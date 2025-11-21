# Langgraph
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

# Typing
from typing import TypedDict, Optional, Annotated, Sequence, List
from pydantic import BaseModel

# configuration and utilities and standard lbraries
from config import API_KEY
from utils import examples_content, cheatsheet_content, ShellSession, init_env_and_log_offsets, read_new_logs
import subprocess
import time
import random
import math
import shlex
import re

# additional tool code
from new_vim_agent import run_file_edit_agent

# global variables
global_session: ShellSession | None = None

# ---------- State Class ----------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    history_summary: Optional[str]
    summarized_upto: int  # how many messages are already summarized


# ---------- Helpers ----------
def get_session() -> ShellSession:
    global global_session
    if global_session is None:
        global_session = ShellSession()
        global_session.connect_root_setSentinel()
        init_env_and_log_offsets(global_session)
    return global_session

def cleanup_session() -> None:
    """
    Read remaining logs and close the global ShellSession if it exists.
    Must NOT create a new session.
    """
    global global_session

    if global_session is None:
        return  # nothing to clean up

    try:
        # best effort: don't let log-reading crash the program shutdown
        read_new_logs(global_session)
    except Exception as e:
        print(f"[cleanup_session] Failed to read logs: {e}")

    try:
        global_session.close()
    except Exception as e:
        print(f"[cleanup_session] Failed to close session: {e}")

    global_session = None

def human_delay_for_cmd(cmd: str) -> None:
    return 
    """
    Simulate a more realistic human pause before typing/running a command.
    Humans don't react instantly — they read, think, hesitate, re-read,
    and only then type. Delay scales with command length and includes
    extra human-like randomness.
    """
    base = 1.2                     # humans take longer before acting
    per_char = 0.03                # ~30ms per character typed
    cognitive_delay = random.uniform(0.5, 2.0)  # thinking/hesitation
    jitter = random.gauss(0, 0.7)  # natural inconsistency

    delay = base + per_char * len(cmd) + cognitive_delay + jitter

    # clamp into a slower, believable human range
    delay = max(1.0, min(delay, 10.0))

    print(f"[human_delay_for_cmd] pausing for {delay:.2f}s (thinking/typing)…")
    time.sleep(delay)


def human_delay_for_vim() -> None:
    return 
    """
    Simulate a human preparing for a Vim editing session.
    People read the file, think about changes, scroll, hesitate,
    and often take significantly longer than simple command execution.
    """
    mean = 6.0                    # typical human pauses longer before editing
    std_dev = 3.0                 # very inconsistent
    planning_delay = random.uniform(1.0, 4.0)  # extra time to read/understand
    delay = random.gauss(mean, std_dev) + planning_delay

    # realistic editing-prep delay range
    delay = max(4.0, min(delay, 20.0))

    print(f"[human_delay_for_vim] pausing for {delay:.2f}s (reading/editing)…")
    time.sleep(delay)

def build_model_messages(
    state_messages: Sequence[BaseMessage],
    max_history: int = 25,
) -> List[BaseMessage]:
    all_msgs = list(state_messages)

    # 1) Find the first HumanMessage (root user task)
    root_human = None
    for m in all_msgs:
        if isinstance(m, HumanMessage):
            root_human = m
            break

    # 2) Take the last `max_history` messages as a raw window
    recent = all_msgs[-max_history:]

    # 3) Collect tool_call_ids from AI messages in this window
    valid_tool_ids = set()
    for m in recent:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                # LangChain tool_calls usually look like {"id": "...", "type": "tool_call", "function": {...}}
                tid = tc.get("id") or tc.get("tool_call_id")
                if tid:
                    valid_tool_ids.add(tid)

    # 4) Filter out ToolMessages whose tool_call_id is not present in this window
    filtered: List[BaseMessage] = []
    for m in recent:
        if isinstance(m, ToolMessage):
            tcid = getattr(m, "tool_call_id", None)
            if tcid and tcid not in valid_tool_ids:
                # Orphaned ToolMessage -> drop it
                continue
        filtered.append(m)

    # 5) (Optional but nice) drop *leading* ToolMessages even after filtering,
    #    to avoid starting context with a random tool result.
    while filtered and isinstance(filtered[0], ToolMessage):
        filtered.pop(0)

    # 6) Make sure the root human is present (at the front)
    if root_human is not None and root_human not in filtered:
        msgs = [root_human] + filtered
    else:
        msgs = filtered

    return msgs


def maybe_summarize_history(state: AgentState, threshold: int = 25) -> dict:
    """
    Optionally update the running summary.

    Returns a dict with updated 'history_summary' and 'summarized_upto',
    or {} if we decide not to summarize this step.
    """
    messages = list(state["messages"])
    old_summary = state.get("history_summary")
    summarized_upto = state.get("summarized_upto", 0)

    # Only summarize if there are enough *new* messages
    new_count = len(messages) - summarized_upto
    if new_count < threshold:
        return {}

    # Take only the new chunk
    new_chunk = messages[summarized_upto:]

    # Turn messages into a simple transcript
    transcript_lines = []
    for m in new_chunk:
        if isinstance(m, HumanMessage):
            role = "USER"
            continue # not needed
        elif isinstance(m, AIMessage):
            role = "ASSISTANT"
        elif isinstance(m, ToolMessage):
            role = f"TOOL[{m.name}]"
        else:
            role = m.__class__.__name__
        transcript_lines.append(f"{role}: {m.content}")
    transcript = "\n".join(transcript_lines)

    # Build prompt for summarizer
    prompt: List[BaseMessage] = [
        SystemMessage(
            content=(
                "Summarize the recent web-administration activity concisely.\n"
                "Focus only on what the AI agent must remember in order to continue the task correctly:\n"
                "- commands that were executed\n"
                "- files that were inspected or edited\n"
                "- services that were checked or changed\n"
                "- important errors, warnings, or system states\n\n"
                "Keep it super super short, factual, and useful for future reasoning."
            )
        )
    ]

    if old_summary:
        prompt.append(
            HumanMessage(
                content=(
                    "Here is the existing summary of earlier steps:\n"
                    f"{old_summary}\n\n"
                    "Update this summary based on the following new turns:\n"
                    f"{transcript}"
                )
            )
        )
    else:
        prompt.append(
            HumanMessage(
                content=(
                    "Create a concise summary of the following:\n"
                    f"{transcript}"
                )
            )
        )

    summary_msg = summary_model.invoke(prompt)
    new_summary = summary_msg.content.strip()

    ### DEBUG
    print("Debugging from maybe_summarize:")
    print("summarized history:", new_summary)
    print("summarized_upto:", len(messages))


    return {
        "history_summary": new_summary,
        "summarized_upto": len(messages),
    }




# ---------- Tools ----------
'''
@tool
def read_file(filename: str, search: str) -> str:
    """
    Read specific parts of a file based on a regex or search phrase.

    This is the preferred way to inspect file content before using use_vim.
    It does NOT return the whole file, only lines matching the search query.

    Args:
        filename (str): Path to the file.
        search (str): Regex or plain text (case-insensitive).
                      If empty, a short head/tail preview is shown.

    Returns:
        str: Matching lines with line numbers (or preview if no search).
    """

    print("------------------------- Entered read_file -------------------------")
    print("filename:", filename)
    print("search:", search)

    # Human-like delay (just for realism)
    human_delay_for_cmd(f"grep {search} {filename}")

    session = get_session()

    # If search is empty → return a preview instead of full file
    if not search.strip():
        # leave filename unquoted so ~ and $HOME are expanded by the remote shell
        cmd = f"(head -n 30 {filename}; echo '---'; tail -n 30 {filename})"
    else:
        # keep it simple: put the pattern in single quotes
        # this makes spaces work and keeps the regex intact
        pattern = search.strip().replace("'", r"'\''")
        cmd = f"grep -n -i -E '{pattern}' {filename} || echo '[NO MATCHES FOUND]'"

    raw_result = session.run_cmd(cmd) or ""

    MAX_CHARS = 400
    if len(raw_result) > MAX_CHARS:
        snippet = raw_result[:MAX_CHARS]
        return (
            f"[OUTPUT TRUNCATED: showing first {MAX_CHARS} of {len(raw_result)} characters]\n"
            + snippet.strip()
        )

    if not raw_result.strip():
        return "[NO OUTPUT]"

    return raw_result.strip()
'''

@tool
def read_file(filename: str) -> str:
    """
    Read and return the full content of a file.

    Behavior:
      - Returns the entire file text.
      - If content exceeds 4000 characters, output is truncated to the first 4000.
      - Returns an error message if the file cannot be read.

    Args:
        filename (str): Path to the file.

    Returns:
        str: Full or truncated file content.
    """

    print("------------------------- Entered read_file -------------------------")
    print("filename:", filename)

    # Human-style delay (reading a file, thinking about it, etc.)
    human_delay_for_cmd(f"cat {filename}")

    session = get_session()

    # We intentionally do NOT quote the filename to preserve ~ expansion.
    cmd = f"cat {filename} 2>/dev/null"

    raw = session.run_cmd(cmd)
    if raw is None or raw.strip() == "":
        return f"[ERROR: Cannot read file '{filename}' or file is empty]"

    MAX_CHARS = 4000

    cleaned = raw.rstrip("\n")
    if len(cleaned) > MAX_CHARS:
        return (
            f"[OUTPUT TRUNCATED: showing first {MAX_CHARS} of {len(cleaned)} characters]\n"
            f"(Tip: use next_command with grep or search patterns for targeted inspection)\n\n"
            + cleaned[:MAX_CHARS]
        )

    return cleaned



@tool
def next_command(cmd: str) -> str:
    """
    Execute a single, non-interactive shell command on the Ubuntu 24.04 host (root).

    Args:
        cmd (str): The full command to run (exactly ONE line, e.g. "ls -la /var/log").

    Returns:
        str: The trimmed stdout/stderr result of the command.

    Usage guidelines:
        - Avoid destructive operations.
        - One command per call; keep it ONE line.
        - Keep output short; use pipes like grep/head/tail for large logs.
        - No interactive/full-screen tools (e.g., less, top in interactive mode).

    Environment:
        - You are root via SSH (sudo -i).
        - The following env vars are assumed for non-paged output:
            export SYSTEMD_URLIFY=0
            export SYSTEMD_PAGER=
            export SYSTEMD_COLORS=0

    Nextcloud context:
        - Install path: /var/www/nextcloud
        - Run occ as: /usr/bin/php /var/www/nextcloud/occ

    Logs:
        - Apache error: /var/log/apache2/nextcloud.local-error.log
        - Apache access: /var/log/apache2/nextcloud.local-access.log

    System tools available:
        - curl (HTTP), net-tools, top (batch mode: "top -b -n 1")
    """
    print("------------------------- Entered next_command -------------------------")
    print("cmd:", cmd)

    # --- Human-like delay ---
    human_delay_for_cmd(cmd)

    MAX_CHARS = 400
    session = get_session()
    raw_result = session.run_cmd(cmd) or ""

    if not raw_result.strip():
        return "[NO OUTPUT]"

    if len(raw_result) > MAX_CHARS:
        snippet = raw_result[-MAX_CHARS:]
        return (
            f"[OUTPUT TRUNCATED: showing last {MAX_CHARS} of {len(raw_result)} characters]\n"
            + snippet.strip()
        )

    return raw_result.strip()


@tool
def use_browser(query: str) -> str:
    """
    Automate browser interactions on nextcloud.local.
    The **minimum required action for every request is a full login attempt**.
    
    Args:
        query (str): A short natural-language instruction or command sequence for the browser tool.

    Returns:
        str: A brief summary of the browser tasks performed.
    """

    print("------------------------- Entered use_browser -------------------------")
    print("Query:", query)

    cmd = ["python", "client_openai.py", "--playwright"]
    
    # Run the command, pass the query via stdin, and capture the output
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="."  # client path
    )

    try:
        stdout, stderr = process.communicate(query + "\n", timeout=300)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.communicate()
        except Exception:
            pass
        return "Browser tool error: timed out while executing browser automation."


    if process.returncode != 0:
        return f"Browser tool error: {stderr.strip()}"


    clean_output = "\n".join(
        line for line in stdout.splitlines() if not line.startswith("Query:")
    ).strip()

    return clean_output


@tool
def use_vim(filename: str, query: str) -> str:
    """
    Edit a file using Vim based strictly on the file’s current content and the
    explicit instructions in `query`.

    The tool has **no external knowledge** beyond what is shown in the file and
    must **not invent or guess** values. Every change must be fully specified
    in the query (e.g. “replace X with Y”, “delete line containing Z”, 
    “uncomment this exact line”, etc.).

    Examples of valid instructions:
      - "Replace 'foo' with 'bar' in the config array."
      - "Uncomment the line 'overwrite.cli.url'."
      - "Insert the text ... above the closing bracket."

    Examples of invalid instructions:
      - "Fix the incorrect settings."
      - "Set this to the right value."

    Returns:
        A short report of applied changes, or an explanation if no safe,
        explicit edits could be made.
    """

    print("------------------------- Entered use_vim -------------------------")
    print("filename:", filename)
    print("Query:", query)


    # --- Human-like delay ---
    human_delay_for_vim()

    session = get_session()
    try:
        session.start_vim(filename=filename)
        file_content = session.print_file_vim()
        result = run_file_edit_agent(query=query, file_content=file_content)
        updated_file = result["updated_file"]
        explanation = result["explanation"]
        session.overwrite_vim(updated_file)
        session.end_vim()
        return explanation
    except Exception as e:
        # try to escape Vim and restore shell
        try:
            session._vim_escape_hatch()
        except Exception:
            # don't let a failed escape completely hide the original error
            pass

        return (
            "Error occurred during use_vim tool! "
            "You can fall back to a one-line edit via next_command (e.g., using sed)."
        )
    

@tool
def terminate(summary: str) -> str:
    """
    Signal that the task is fully complete.

    `summary` should briefly describe:
      - what you changed
      - current status
    """
    return summary


tools = [next_command, use_browser, read_file, use_vim, terminate]

# ---------- LLM client ----------
summary_model = ChatOpenAI(
    model="gpt-4o-mini",  # cheap summarizer
    api_key=API_KEY,
    temperature=0.0,
)

model = ChatOpenAI(model="gpt-4o", api_key=API_KEY, temperature=0.1).bind_tools(tools=tools)
    #tool_choice={"type": "function", "function": {"name": "it_is_enough"}}  # <-- correct shape


# ---------- Nodes ----------
def decision_node(state: AgentState) -> AgentState:

    # 1) Maybe update the running summary
    summary_updates = maybe_summarize_history(state)
    history_summary = summary_updates.get("history_summary", state.get("history_summary"))
    summarized_upto = summary_updates.get("summarized_upto", state.get("summarized_upto", 0))

    system_prompt_parts = [
        "Reference:\n",
        f"Examples (Admin Routines):\n{examples_content}\n\n",
        f"Cheat Sheet (occ):\n{cheatsheet_content}\n\n",
        "You are a linux web-administrator for Ubuntu 24.04 (LAMP, Nextcloud PHP 8.3.6).\n\n",
        "=== Agent tools ===\n",
        "1. `next_command(cmd: str)` — run a shell command.\n"
        "2. `use_browser(query: str)` — interact with the web UI and verify problems (prefer over curl).\n"
        "3. `use_vim(filename: str, query: str)` — edit files.\n",
        "4. `read_file(filename: str)` — read the full content of a file (auto-truncated if very large).\n"
        #"After performing changes, verify it with an appropriate command (e.g., cat the file, check service status)\n"
        "=== Agent Behaviour Rules ===\n"
        "- Do NOT ask the user any questions. There is NO interactive user.\n"
        "- When you reach the end of the task, call the `terminate` tool with a brief summary of your actions and the final state.\n"
    ]

    system_prompt = "".join(system_prompt_parts)

    # 3) Build recent history window (smaller)
    chat_history = build_model_messages(state["messages"], max_history=12)

    # 4) Assemble messages for main model
    model_messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

    if history_summary:
        model_messages.append(
            SystemMessage(
                content=(
                    "Summary of previous steps (do not repeat these actions unless needed):\n"
                    f"{history_summary}"
                )
            )
        )

    model_messages.extend(chat_history)

    # 5) Call main model
    response = model.invoke(model_messages)

    # 6) Return new AI message + summary fields
    return {
        "messages": [response],
        "history_summary": history_summary,
        "summarized_upto": summarized_upto,
    }


# ---------- Build graph ----------
graph = StateGraph(AgentState)
graph.add_node("decision_node", decision_node)
tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)
graph.add_edge(START, "decision_node")
    
def route_decision(state: AgentState) -> str:
    last = state["messages"][-1]
    # If the LLM asked to call a tool, go to the ToolNode
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "__tool__"
    return "__decide__"

graph.add_conditional_edges(
    "decision_node",
    route_decision,
    path_map={
        "__tool__": "tool_node",
        "__decide__": "decision_node",
    },
)

def route_after_tool(state: AgentState) -> str:
    # Find the last ToolMessage (the most recent tool result)
    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    last_tool = tool_msgs[-1] if tool_msgs else None

    if last_tool and last_tool.name == "terminate":
        return "__end__"
    return "__decide__"

graph.add_conditional_edges(
    "tool_node",
    route_after_tool,
    path_map={
        "__end__": END,
        "__decide__": "decision_node",
    },
)

# ---------- Compile & run ----------
app = graph.compile()


if __name__ == "__main__":

    result = None   # <-- ensures finally block can access it

    try:
        result = app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="There is something wrong with nextcloud.Fix it!"
                    )
                ],
                "history_summary": None,
                "summarized_upto": 0,
            },
            config={"recursion_limit": 200},
        )

    finally:
        # ---- PRINT OUTPUT SAFELY ----
        print("Output:")
        if result is not None:
            for message in result["messages"]:
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()
        else:
            print("[NO RESULT — the agent crashed before producing output]")

        # ---- CLEAN UP SESSION ----
        cleanup_session()