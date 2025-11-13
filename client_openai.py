import asyncio
import sys
import os
import re
import json
from typing import Optional, Any, Dict, List
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI  # OpenAI SDK
from dotenv import load_dotenv

# ---------- Parameter section! ----------
DEBUG_FLAG = False
MODEL = "gpt-4.1"
TEMPERATURE = 0.1

# --- Load .env BEFORE reading env vars ---
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", MODEL)

# ---------- Token + snapshot helpers (unchanged) ----------

def _est_tokens(text: str) -> int:
    """Very rough token estimate: ~1 token per 4 chars."""
    return max(1, len(text) // 4)

def _truncate_text(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + " …"

def _extract_yaml_block(full_text: str) -> str | None:
    start = full_text.find("```yaml")
    if start == -1:
        return None
    end = full_text.find("```", start + 7)
    if end == -1:
        return None
    return full_text[start + 7:end].strip()

_KEEP_ROLES_RE = re.compile(
    r'\b(heading|link|button|combobox|search|navigation|main|table|row|cell|'
    r'list(item)?|figure|paragraph|)\b', # optionally to add: img|generic|div
    re.IGNORECASE
)

def _compact_snapshot_yaml(
    yaml_text: str,
    keep_max_lines: int = 1200, # param
    line_char_cap: int = 280 # param
) -> str:
    lines = yaml_text.splitlines()
    kept: List[str] = []
    for ln in lines:
        if _KEEP_ROLES_RE.search(ln) or len(kept) < 20:
            kept.append(_truncate_text(ln, line_char_cap))
        if len(kept) >= keep_max_lines:
            break
    return "\n".join(kept)

def _summarize_snapshot_text(full_text: str, max_chars: int = 3000) -> str: # param
    """Return URL/Title header + compacted YAML section."""
    url = None
    title = None
    for line in full_text.splitlines():
        if line.startswith("- Page URL:"):
            url = line.split(":", 1)[1].strip()
        elif line.startswith("- Page Title:"):
            title = line.split(":", 1)[1].strip()
        if url and title:
            break

    yaml_block = _extract_yaml_block(full_text)
    compact_yaml = _compact_snapshot_yaml(yaml_block) if yaml_block else None

    head = []
    if url: head.append(f"URL: {url}")
    if title: head.append(f"Title: {title}")
    head_text = "\n".join(head)

    body = f"```yaml\n{compact_yaml}\n```" if compact_yaml else ""
    combined = (head_text + ("\n\n" if head_text and body else "") + body).strip()
    return _truncate_text(combined, max_chars)

def _mcp_content_to_text_blocks(content_list: List[Any], include_full_snapshots: bool = False) -> List[str]:
    """
    Similar to the Anthropic normalizer but return plain strings for OpenAI tool replies.
    We'll JSON-serialize these when sending back via the 'tool' role message.
    """
    out: List[str] = []

    # helper: extract the "### Result ..." blob if present (fenced or unfenced)
    def _extract_result_blob(txt: str) -> str | None:
        m = re.search(
            r"### Result\s+```?json?\s*(.*?)```|### Result\s*([\s\S]*?)\n###",
            txt,
            re.S
        )
        if not m:
            return None
        return (m.group(1) or m.group(2) or "").strip()

    for item in content_list or []:
        # normalize to a text string (txt) if it's a "text" item
        txt = None
        if hasattr(item, "type") and getattr(item, "type") == "text" and hasattr(item, "text"):
            txt = getattr(item, "text") or ""
        elif isinstance(item, dict) and item.get("type") == "text":
            txt = item.get("text", "") or ""

        if txt is not None:
            # If the tool text contains a Page Snapshot, prefer the actual Result first
            if "Page Snapshot:" in txt:
                blob = _extract_result_blob(txt)
                if blob:
                    # We got a real result — only keep that
                    out.append(_truncate_text(blob, 8000))
                    continue  # skip snapshot entirely
                # else, no result → keep snapshot (full or compact)
                if include_full_snapshots:
                    out.append(_truncate_text(txt, 8000))
                else:
                    out.append(_summarize_snapshot_text(txt))
            else:
                # no snapshot: just add the text (truncated)
                out.append(_truncate_text(txt, 8000))
        else:
            # non-text item: stringify safely
            out.append(_truncate_text(str(item), 4000))

    if not out:
        out = ["(empty tool result)"]
    return out



def _trim_messages_to_budget(messages: List[Dict[str, Any]], max_input_tokens: int = 6000) -> List[Dict[str, Any]]:
    """
    Keep most recent content while staying under a rough token budget.
    - Always keep system messages (untrimmed, at the front).
    - Treat an assistant tool-call turn + its following tool replies as an atomic bundle.
    - Trim from the end by whole bundles so tool messages always have their parent.
    """
    # 0) keep system messages untouched
    system_user_msgs = [m for m in messages if m.get("role") == "system" or m.get("role") == "user"]
    non_system = [m for m in messages if m.get("role") not in {"system", "user"}]

    # helper: rough token count of one message
    def _msg_tokens(msg: Dict[str, Any]) -> int:
        parts: List[str] = []
        c = msg.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for blk in c:
                if isinstance(blk, dict) and "text" in blk:
                    parts.append(str(blk["text"]))
                else:
                    parts.append(str(blk))
        elif isinstance(c, dict):
            parts.append(str(c))
        # include a small header overhead
        return sum(_est_tokens(p) for p in parts) + 12

    # 1) Build bundles (oldest -> newest)
    bundles: List[List[Dict[str, Any]]] = []
    i = 0
    n = len(non_system)
    while i < n:
        m = non_system[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            # assistant tool call bundle
            bundle = [m]
            # collect the set of tool_call_ids
            ids = set()
            for tc in m.get("tool_calls", []) or []:
                # tc can be dict or pydantic-like; handle both
                if isinstance(tc, dict):
                    ids.add(tc.get("id"))
                else:
                    ids.add(getattr(tc, "id", None))
            i += 1
            # attach subsequent tool messages that reference those ids
            while i < n and non_system[i].get("role") == "tool" and (not ids or non_system[i].get("tool_call_id") in ids):
                bundle.append(non_system[i])
                i += 1
            bundles.append(bundle)
        else:
            # single-message bundle
            bundles.append([m])
            i += 1

    # 2) Trim from the end by bundles
    kept: List[List[Dict[str, Any]]] = []
    total_tokens = 0
    for bundle in reversed(bundles):
        bundle_tokens = sum(_msg_tokens(bm) for bm in bundle)
        if kept and total_tokens + bundle_tokens > max_input_tokens:
            break
        kept.append(bundle)
        total_tokens += bundle_tokens

    # 3) Reassemble: system + kept bundles (restore chronological order)
    kept = list(reversed(kept))
    flat_kept = [m for bundle in kept for m in bundle]
    return system_user_msgs + flat_kept

SYSTEM_PROMPT = (
    "You are an automated web assistant restricted to ONLY interact with the website 'nextcloud.local'. "
    "For login use the following credentials:\n"
    "  Username: admin\n"
    "  Password: changeme\n"
    "Perform the assigned task using as few interactions as possible — only the minimum necessary steps. "
    "Do not continue indefinitely or repeat actions. "
    "After completing the task, produce a concise plain-text report of what happened and stop. "
    "Do not ask the user for follow-up steps."
)

# ----------------- Client (OpenAI) -----------------

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI()  # uses OPENAI_API_KEY from env
        self.stdio = None
        self.write = None

    async def connect_via_command(self, command: str, args: list[str] | None = None, env: dict | None = None):
        server_params = StdioServerParameters(command=command, args=args or [], env=env)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        resp = await self.session.list_tools()
        if DEBUG_FLAG:        
            print("\nConnected to server with tools:", [t.name for t in (resp.tools or [])])

    async def connect_to_server(self, server_script_path: str):
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        command = sys.executable if is_python else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        resp = await self.session.list_tools()
        if DEBUG_FLAG:
            print("\nConnected to server with tools:", [t.name for t in (resp.tools or [])])

    def _mcp_tools_to_openai(self, tools_resp) -> List[Dict[str, Any]]:
        """Convert MCP tool descriptions to OpenAI function-calling schema."""
        out = []
        for t in (tools_resp.tools or []):
            out.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema or {"type": "object", "properties": {}}
                }
            })
        return out

    async def process_query(self, query: str) -> str:
        # 1) Build OpenAI tools list
        resp = await self.session.list_tools()
        openai_tools = self._mcp_tools_to_openai(resp)

        # 2) Start messages with a system prompt (OpenAI style)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        final_text: List[str] = []

        for _ in range(20):

            # Trim before each call
            messages = _trim_messages_to_budget(messages, max_input_tokens=6000)

            completion = self.openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=openai_tools if openai_tools else None,
                tool_choice="auto",
                temperature=TEMPERATURE,
            )

            msg = completion.choices[0].message
            tool_calls = msg.tool_calls or []

            # If no tool calls, we got a plain answer
            if not tool_calls:
                if msg.content:
                    final_text.append(msg.content)
                break

            # Append assistant message containing the tool_calls (required context)
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in tool_calls],
            })

            # Execute each requested tool in order and send results back
            for call in tool_calls:
                name = call.function.name
                args = {}
                if call.function.arguments:
                    try:
                        args = json.loads(call.function.arguments)
                    except Exception:
                        # if the model produced invalid JSON, send the raw string
                        args = {"__raw_arguments__": call.function.arguments}

                if DEBUG_FLAG:
                    print(f"\n→ Calling tool '{name}' with args: {args}")

                try:
                    result = await self.session.call_tool(name, args)
            
                    if DEBUG_FLAG:
                        print("\n\n")
                        print("DEBUGGING")
                        print("Current-Tool-Result: result.content:", result.content)
                        print("\n\n")

                    text_blocks = _mcp_content_to_text_blocks(result.content, include_full_snapshots=False)
                    # Serialize to a compact JSON string for the tool message
                    tool_content = json.dumps(text_blocks, ensure_ascii=False)
                except Exception as e:
                    tool_content = json.dumps([f"Tool error: {e}"], ensure_ascii=False)

                # Tool result goes in a 'tool' role message linked by tool_call_id
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": tool_content,
                })

        return "\n".join(final_text) if final_text else "(no text response)"

    async def chat_loop(self):
        while True:
            try:
                q = input("\nQuery: ").strip()
                if q.lower() in {"quit", "exit"}:
                    break
                resp = await self.process_query(q)
                print("\n" + resp)
                # only loop in debug mode
                if not DEBUG_FLAG:
                    break
            except (KeyboardInterrupt, EOFError):
                print("\nExiting…")
                break
            except Exception as e:
                print(f"\nError: {e}")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    client = MCPClient()
    try:
        if len(sys.argv) == 2 and sys.argv[1] == "--playwright":
            await client.connect_via_command(
                "npx",
                ["@playwright/mcp@latest",  "--isolated", "--allowed-hosts", "nextcloud.local", "--browser", "chromium", "--no-sandbox"],
                env={**os.environ}
            )
        elif len(sys.argv) == 2:
            await client.connect_to_server(sys.argv[1])
        else:
            print("Usage: uv run client.py --playwright | <path_to_server_script>")
            sys.exit(1)

        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    #uv run client_openai.py --playwright
    asyncio.run(main())
