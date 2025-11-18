import asyncio
import os
import sys
import json
import re
import subprocess
from typing import Optional, Any, Dict, List
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv

# -------------------- Config --------------------
DEBUG = True
DEFAULT_SOCKET = "/tmp/nvim"   # your local UNIX socket (socat bridge)
MAX_TOOL_STEPS = 40
TEMPERATURE = 0.1
DEFAULT_MODEL = "gpt-4.1"  # or your preferred model
MAX_BLOCK = 400


SYSTEM_PROMPT = (
    "You are a precise Neovim automation agent, currently operating inside an active Neovim session with a file already opened.\n"
    "Goal: complete the user's request with the minimum necessary actions using the available tools.\n"
    "Do not loop or repeat actions unnecessarily.\n"
    "When you believe the task is complete, return a concise plain-text summary of what you did and stop.\n"
    "Guidelines:\n"
    " - The current file is already open in Neovim.\n"
    " - Use 'vim_command' for normal or Ex-mode commands.\n"
    " - Avoid destructive actions unless explicitly requested.\n"
    " - Do not ask the user for follow-up steps.\n"
    "\n"
    "Conditional Edits:\n"
    " - For any conditional request (e.g., 'if something then change it'), ALWAYS:\n"
    "     1) Read the current value from the buffer.\n"
    "     2) Evaluate the condition.\n"
    "     3) Apply the change only if the condition is true.\n"
    " - Never assume values; always extract them.\n"
    "\n"
    "Validation:\n"
    " - After performing edits, ALWAYS re-read the affected line(s) to confirm that the expected change has been applied.\n"
    "\n"
    "- Searching: Try multiple simple searches first (section header -> key name). Don't give up after one failed pattern.\n"
    "- Editing: Make minimal changes only. Do NOT alter formatting or unrelated lines, and never claim 'no changes' if anything was modified.\n"
    "\n"
    "Example:\n"
    "   :keepjumps /timeout_seconds/\n"
    "   :let v = str2nr(matchstr(getline('.'), '\\d\\+'))\n"
    "   :if v < 4 | call setline('.', '  timeout_seconds: 25') | endif\n"
    "   :keepjumps /timeout_seconds/  \" validation step\n"
)


# -------------------- Env --------------------
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
# -------------------- Small helpers --------------------
def _unix_socket_exists(path: str) -> bool:
    try:
        return subprocess.run(["bash", "-lc", f'[[ -S "{path}" ]]'], check=False).returncode == 0
    except Exception:
        return False


def _trim_messages_to_budget(messages: List[Dict[str, Any]], max_input_tokens: int = 2000) -> List[Dict[str, Any]]:
    """
    Keep most recent content while staying under a rough token budget.
    - Always keep ALL system messages (untrimmed, at the front).
    - Always keep the most recent user message.
    - Treat each assistant tool-call turn + its following tool replies as an atomic bundle.
    - Trim from the end by whole bundles so tool messages always have their parent.
    - If the most recent bundle alone exceeds the budget, we still keep it (better to keep the latest step complete).
    """
    # --- helpers ---
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
        return sum(max(1, len(p) // 4) for p in parts) + 12  # ~1 tok / 4 chars + header fudge

    # --- split into head (system + last user) and the rest ---
    system_msgs = [m for m in messages if m.get("role") == "system"]
    user_msgs = [m for m in messages if m.get("role") == "user"]
    last_user = user_msgs[-1] if user_msgs else None

    # everything else in chronological order, excluding system + last_user (we'll re-add latter)
    core: List[Dict[str, Any]] = []
    for m in messages:
        if m in system_msgs:
            continue
        if last_user is not None and m is last_user:
            continue
        core.append(m)

    # --- build bundles from 'core' (oldest -> newest) ---
    bundles: List[List[Dict[str, Any]]] = []
    i = 0
    n = len(core)
    while i < n:
        m = core[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            bundle = [m]
            ids = set()
            for tc in (m.get("tool_calls") or []):
                if isinstance(tc, dict):
                    ids.add(tc.get("id"))
                else:
                    ids.add(getattr(tc, "id", None))
            i += 1
            # attach following tool messages that match those ids
            while i < n and core[i].get("role") == "tool" and (not ids or core[i].get("tool_call_id") in ids):
                bundle.append(core[i])
                i += 1
            bundles.append(bundle)
        else:
            bundles.append([m])
            i += 1

    # --- assemble under budget, trimming from the end (newest first) ---
    kept_bundles: List[List[Dict[str, Any]]] = []
    head_tokens = sum(_msg_tokens(m) for m in system_msgs) + ( _msg_tokens(last_user) if last_user else 0 )
    total = head_tokens

    for idx, bundle in enumerate(reversed(bundles)):
        bundle_tokens = sum(_msg_tokens(bm) for bm in bundle)
        if kept_bundles:
            if total + bundle_tokens > max_input_tokens:
                break
        else:
            # Ensure we keep at least the newest bundle even if it alone exceeds the budget
            # so the model has the latest tool context intact.
            pass
        kept_bundles.append(bundle)
        total += bundle_tokens

    kept_bundles.reverse()  # restore chronological order

    # --- final reassembly: system + last user + kept bundles ---
    flat = []
    flat.extend(system_msgs)
    if last_user:
        flat.append(last_user)
    for b in kept_bundles:
        flat.extend(b)
    return flat


def _content_items_to_text_blocks(content_list: List[Any]) -> List[str]:
    """
    Normalize MCP tool content into safe, short text blocks.
    Hard truncate individual blocks to avoid blowing up the token budget.
    """

    out: List[str] = []

    for item in content_list or []:
        # Extract text
        if isinstance(item, dict) and item.get("type") == "text":
            text = str(item.get("text", ""))
        elif hasattr(item, "type") and getattr(item, "type") == "text":
            text = str(getattr(item, "text", ""))
        else:
            text = str(item)

        # Truncate + add marker
        if len(text) > MAX_BLOCK:
            text = text[:MAX_BLOCK] + " …[truncated]"

        out.append(text)

    return out or ["(empty tool result)"]

# -------------------- Client --------------------
class NeovimAgentClient:
    def __init__(self, socket_path: str = DEFAULT_SOCKET, allow_shell: str = "false"):
        self.socket_path = socket_path
        self.allow_shell = allow_shell.lower()
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.stdio = None
        self.write = None
        self.openai = OpenAI()  # requires OPENAI_API_KEY env

    async def connect(self):
        # Preflight: ensure local UNIX socket exists (created by your socat bridge)
        if not _unix_socket_exists(self.socket_path):
            raise RuntimeError(
                f"Expected UNIX socket at {self.socket_path}.\n"
                f"Start tunnel + bridge first, e.g.:\n"
                f"  ssh -f -N -L 6666:127.0.0.1:6666 server\n"
                f"  socat UNIX-LISTEN:{self.socket_path},fork,unlink-early TCP:127.0.0.1:6666"
            )

        env = {
            **os.environ,
            "NVIM_SOCKET_PATH": self.socket_path,
            "ALLOW_SHELL_COMMANDS": "true" if self.allow_shell == "true" else "false",
        }
        params = StdioServerParameters(
            command="npx",
            args=["-y", "mcp-neovim-server"],
            env=env,
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        if DEBUG:
            tools = await self.session.list_tools()
            print("Connected. Tools:", [t.name for t in (tools.tools or [])])


    async def _current_file(self) -> Optional[str]:
        """Return absolute path of the currently active buffer, or None."""
        try:
            res = await self.session.call_tool("vim_command", {"command": ":echo expand('%:p')"})
            blocks = _content_items_to_text_blocks(res.content)
            # First block should contain the echo result
            val = (blocks[0] if blocks else "").strip()
            return val or None
        except Exception:
            return None

    async def close(self):
        await self.exit_stack.aclose()

    # ---- LLM orchestration ----
    def _mcp_tools_to_openai(self, tools_resp) -> List[Dict[str, Any]]:
        out = []
        for t in (tools_resp.tools or []):
            if t.name == "vim_buffer":
                continue  # <-- skip it completely
            out.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema or {"type": "object", "properties": {}},
                },
            })
        return out

    async def run_query(self, user_query: str, model: str = OPENAI_MODEL) -> str:
        # 1) Discover tools
        tools_resp = await self.session.list_tools()
        openai_tools = self._mcp_tools_to_openai(tools_resp)

        # 1.5) Capture current file path from the running Neovim
        current_file = await self._current_file()
        if DEBUG:
            print("Current file detected:", current_file or "(unknown)")

        # 2) Seed conversation
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if current_file:
            messages.append({"role": "system", "content": f"(Context: current file is {current_file})"})
        messages.append({"role": "user", "content": user_query})
        final_text: List[str] = []

        for step in range(MAX_TOOL_STEPS):
            messages = _trim_messages_to_budget(messages, max_input_tokens=2000)

            completion = self.openai.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_tools if openai_tools else None,
                tool_choice="auto",
                temperature=TEMPERATURE,
            )
            msg = completion.choices[0].message
            tool_calls = msg.tool_calls or []

            if DEBUG:
                print(f"\n[STEP {step+1}] Assistant message:", msg.content or "(tool calls)")

            # If model is done (no tool calls), collect final text and stop
            if not tool_calls:
                if msg.content:
                    final_text.append(msg.content)
                break

            # 3) Add assistant message with tool_calls to the transcript
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in tool_calls],
            })

            # 4) Execute tools in sequence and feed results back
            for call in tool_calls:
                name = call.function.name
                raw_args = call.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except Exception:
                    args = {"__raw_arguments__": raw_args}

                try:
                    args = json.loads(raw_args)

                except Exception as e:
                    if DEBUG:
                        print("\n[DEBUG] INVALID TOOL ARGUMENTS")
                        print(f"Tool: {name}")
                        print(f"Error: {e}")
                        print(f"Raw args (first 300 chars):\n{raw_args[:300]}")
                        print("-" * 60)

                    # Tell the model the call was malformed, without dumping massive blobs
                    tool_error = (
                        f"Invalid JSON for tool '{name}': {e}. "
                        f"First 200 chars of raw args: {raw_args[:200]}..."
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps([tool_error], ensure_ascii=False),
                    })

                    continue  # skip calling the tool

                if DEBUG:
                    print(f"→ Calling tool '{name}' with args: {args}")

                try:
                    result = await self.session.call_tool(name, args)
                    blocks = _content_items_to_text_blocks(result.content)
                    tool_reply = json.dumps(blocks, ensure_ascii=False)

                    # --- NEW DEBUG LOG: pretty-print tool output ---
                    if DEBUG:
                        print(f"[DEBUG] Tool '{name}' output blocks:")
                        for b in blocks:
                            print("    ", b)
                        print("-" * 60)

                except Exception as e:
                    tool_reply = json.dumps([f"Tool error: {e}"], ensure_ascii=False)

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": tool_reply,
                })

        return "\n".join(final_text) if final_text else "(no text response)"

# -------------------- CLI --------------------
async def main():
    socket = os.getenv("NVIM_SOCKET_PATH", DEFAULT_SOCKET)
    allow_shell = os.getenv("ALLOW_SHELL_COMMANDS", "true")

    user_query = input("Neovim agent query> ").strip()
    if not user_query:
        print("No query provided. Exiting.")
        return

    client = NeovimAgentClient(socket_path=socket, allow_shell=allow_shell)
    try:
        await client.connect()
        result = await client.run_query(user_query)
        print("\n=== Agent Summary ===\n" + result)
    finally:
        await client.close()



from tunnel_bridge import setup_remote_nvim_bridge


if __name__ == "__main__":
    setup_remote_nvim_bridge()
    asyncio.run(main())
