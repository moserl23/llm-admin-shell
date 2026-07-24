"""LangGraph runner for autonomous Nextcloud troubleshooting.

The agent alternates between tool use and model decisions, while keeping a
running summary so long trajectories remain tractable without losing state.
"""

# ---------- LangGraph and LLM plumbing ----------
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

# ---------- Typing ----------
from typing import TypedDict, Optional, Annotated, Sequence, List

# ---------- Environment, utilities, and standard library ----------
#from config import API_KEY
from dotenv import load_dotenv
load_dotenv()
from ..utils import examples_content, cheatsheet_content, ShellSession, init_env_and_log_offsets, read_new_logs
import subprocess
import time
import signal
import sys
import random

# ---------- Tool helper ----------
from .vim_agent import run_file_edit_agent

# ---------- Global state ----------
global_session: ShellSession | None = None


# ---------- Hyperparameters / Config ----------
class AgentConfig:

    # Metrik Prio 1: Least Number of Failures / Least Number of Recursions

    # problem description
    problem_prompt = "Files section of nextcloud shows a problem!"    

    # Summarization behavior
    SUMMARY_THRESHOLD = 36          # summarize after this many new messages
    # Chat context window
    MAX_HISTORY_WINDOW = SUMMARY_THRESHOLD         # number of recent messages passed to main model
    # [6, 12, 24, 36, 48, 60]
    # Default: 24
    # Chosen: 36

    # Tool output truncation limits
    READ_FILE_MAX_CHARS = 4000      # Maximum number of characters returned by read_file (excess is truncated)
    # Default: 4000

    NEXT_COMMAND_MAX_CHARS = 500    # Maximum number of characters returned from a single next_command execution
    # [300, 500, 800]
    # Default: 500
    # Chosen: 500

    # Human-like interaction delays
    DELAY_ACTIVE = True           # <-- toggle delay simulation ON/OFF; only necessary for actual logging

    # In Context Learning
    ENABLE_IN_CONTEXT_EXAMPLES = True
    # [True, False]
    # Default: True
    # Chosen: True

    # LangGraph recursion
    RECURSION_LIMIT = 300

    # LLM configuration
    MAIN_MODEL_NAME = "gpt-4.1"
    # Default: gpt-4.1
    MAIN_MODEL_TEMPERATURE = 0.1
    # [0.1, 0.3]
    # Default: 0.1
    # Chosen: 0.1

    SUMMARY_MODEL_NAME = "gpt-4.1-mini"
    SUMMARY_MODEL_TEMPERATURE = 0.1



# ---------- State ----------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    history_summary: Optional[str]
    summarized_upto: int  # how many messages are already summarized
    decision_steps: int


# ---------- Helpers ----------
def get_session() -> ShellSession:
    """Return the singleton root ShellSession, creating and initializing it on first use."""
    global global_session
    if global_session is None:
        global_session = ShellSession()
        global_session.connect_root_setSentinel()
        init_env_and_log_offsets(global_session)
    return global_session

def cleanup_session() -> None:
    """
    Flush any remaining logs and close the shared shell session.

    This is a shutdown-only path and must never create a new session.
    """
    global global_session

    if global_session is None:
        return  # nothing to clean up

    try:
        # Shutdown should remain best-effort even if log collection fails.
        read_new_logs(global_session)
    except Exception as e:
        print(f"[cleanup_session] Failed to read logs: {e}")

    try:
        global_session.close()
    except Exception as e:
        print(f"[cleanup_session] Failed to close session: {e}")

    global_session = None

def human_delay_for_cmd(cmd: str) -> None:
    """
    Simulate a human pause before issuing a shell command.

    The delay scales with command length and is clipped to a plausible range.
    """

    if not AgentConfig.DELAY_ACTIVE:
        return

    base = 3.5
    per_char = 0.06
    cognitive_delay = random.uniform(1.5, 4.0)
    jitter = random.gauss(0, 0.8)

    delay = base + per_char * len(cmd) + cognitive_delay + jitter
    delay = max(5, min(delay, 10.0))

    print(f"[human_delay_for_cmd] pausing for {delay:.2f}s (thinking/typing)…")
    time.sleep(delay)


def human_delay_for_vim() -> None:
    """
    Simulate the longer pause that typically precedes an editing action.

    Editing involves additional reading and planning compared with shell use.
    """

    if not AgentConfig.DELAY_ACTIVE:
        return

    mean = 10.0
    std_dev = 4.0
    planning_delay = random.uniform(2.0, 6.0)

    delay = random.gauss(mean, std_dev) + planning_delay

    delay = max(8.0, min(delay, 20.0))

    print(f"[human_delay_for_vim] pausing for {delay:.2f}s (reading/editing)…")
    time.sleep(delay)

def invoke_with_retry(model, model_messages, max_retries: int = 3):
    """
    Retry a model call when it fails with a transient rate-limit error.

    Non-rate-limit failures are surfaced immediately; successful calls return
    the model response unchanged.
    """
    base_delay = 1.5  # seconds
    for attempt in range(max_retries + 1):
        try:
            return model.invoke(model_messages)

        except Exception as e:
            msg = str(e).lower()

            # Match both provider-native and wrapped LangChain rate-limit errors.
            is_rate_limit = (
                "rate limit" in msg
                or "429" in msg
                or "tpm" in msg
                or "tokens per min" in msg
                or "rate_limit_exceeded" in msg
            )
            if not is_rate_limit:
                raise  # not a transient rate limit

            if attempt >= max_retries:
                raise  # give up

            delay = base_delay * (2 ** attempt)
            delay = min(delay, 15.0)
            delay += random.uniform(0.0, 0.75)

            print(f"[invoke_with_retry] Rate-limited (attempt {attempt+1}/{max_retries}). Sleeping {delay:.2f}s…")
            time.sleep(delay)



def build_model_messages(
    state_messages: Sequence[BaseMessage],
    max_history: int = None,
) -> List[BaseMessage]:
    """
    Build a bounded message window for the main model.

    The root user task is retained, and orphaned tool outputs are removed so
    the truncated history still reflects a valid tool-call sequence.
    """
    all_msgs = list(state_messages)

    # Keep the original task visible even when the working window is truncated.
    root_human = None
    for m in all_msgs:
        if isinstance(m, HumanMessage):
            root_human = m
            break

    recent = all_msgs[-max_history:]

    # Only tool outputs tied to tool calls still in the window should survive.
    valid_tool_ids = set()
    for m in recent:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                tid = tc.get("id") or tc.get("tool_call_id")
                if tid:
                    valid_tool_ids.add(tid)

    filtered: List[BaseMessage] = []
    for m in recent:
        if isinstance(m, ToolMessage):
            tcid = getattr(m, "tool_call_id", None)
            if tcid and tcid not in valid_tool_ids:
                continue
        filtered.append(m)

    # A context window should not begin with an unexplained tool result.
    while filtered and isinstance(filtered[0], ToolMessage):
        filtered.pop(0)

    if root_human is not None and root_human not in filtered:
        msgs = [root_human] + filtered
    else:
        msgs = filtered

    return msgs


def maybe_summarize_history(state: AgentState, threshold: int = None) -> dict:
    """
    Update the running summary when enough new messages have accumulated.

    Summarization is incremental: only unseen messages are compressed, and the
    function returns updated summary metadata or `{}` when no update is needed.
    """
    messages = list(state["messages"])
    old_summary = state.get("history_summary")
    summarized_upto = state.get("summarized_upto", 0)

    # Summarize in chunks to avoid paying the summarization cost every turn.
    new_count = len(messages) - summarized_upto
    if threshold is not None and new_count < threshold:
        return {}

    new_chunk = messages[summarized_upto:]

    # The summarizer operates on a role-tagged transcript rather than raw objects.
    transcript_lines = []
    for m in new_chunk:
        if isinstance(m, HumanMessage):
            role = "USER"
        elif isinstance(m, AIMessage):
            role = "ASSISTANT"
        elif isinstance(m, ToolMessage):
            role = f"TOOL[{m.name}]"
        else:
            role = m.__class__.__name__
        transcript_lines.append(f"{role}: {m.content}")
    transcript = "\n".join(transcript_lines)

    prompt: List[BaseMessage] = [
        SystemMessage(
            content=(
                "You are a summarization module for a web-administration agent.\n"
                "Compress this web-administrator troubleshooting activity into a minimal working state that preserves only what is required to continue.\n"
            )
        )
    ]

    if old_summary:
        prompt.append(
            HumanMessage(
                content=(
                    "Update the running summary.\n"
                    "Merge the PREVIOUS SUMMARY with the NEW ACTIVITY into ONE summary.\n"
                    "This must be a replacement (do not append) and should not significantly increase in length.\n"
                    "PREVIOUS SUMMARY:\n"
                    f"{old_summary}\n\n"
                    "NEW ACTIVITY:\n"
                    f"{transcript}"
                )
            )
        )
    else:
        prompt.append(
            HumanMessage(
                content=(
                    "Create an initial running summary from the activity below.\n"
                    "ACTIVITY:\n"
                    f"{transcript}"
                )
            )
        )


    summary_msg = summary_model.invoke(prompt)
    new_summary = summary_msg.content.strip()

    # Emit the replacement summary for debugging and trajectory inspection.
    print("Current Summary:")
    print("summarized history:", new_summary)
    print("summarized_upto:", len(messages))


    return {
        "history_summary": new_summary,
        "summarized_upto": len(messages),
    }




# ---------- Tools ----------
@tool
def read_file(filename: str) -> str:
    """
    Read a file and return its contents.

    Large files are truncated at the beginning; inspect specific parts with grep or similar tools.
    Returns an error message if the file cannot be read.

    Example:
    - "/etc/myapp/config.yaml"

    """

    print("------------------------- Entered read_file -------------------------")
    print("filename:", filename)

    human_delay_for_cmd(f"cat {filename}")

    session = get_session()

    # Leave `~` unquoted so the shell can expand it for home-directory paths.
    cmd = f"cat {filename} 2>/dev/null"

    raw = session.run_cmd(cmd)
    if raw is None or raw.strip() == "":
        return f"[ERROR: Cannot read file '{filename}' or file is empty]"

    MAX_CHARS = AgentConfig.READ_FILE_MAX_CHARS

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
    Execute exactly one NON-INTERACTIVE shell command as root.

    Do not use interactive or full-screen programs (e.g. vim, less, top in interactive mode).
    Use pipes to keep output short; large output is truncated.
    Returns "[NO OUTPUT]" if nothing is printed.

    Example:
      - "grep -n 'server_url' /etc/myservice/config.yaml"
    """

 
    print("------------------------- Entered next_command -------------------------")
    print("cmd:", cmd)

    human_delay_for_cmd(cmd)

    MAX_CHARS = AgentConfig.NEXT_COMMAND_MAX_CHARS
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
    Automate browser interaction with the Nextcloud web UI.

    Performs the actions described in `query` within the web interface.
    Each invocation is supposed to attempt at least a login to Nextcloud.

    Example:
      - "Log into Nextcloud and check whether the navigation bar is visible."

    Returns a brief summary of the actions performed.
    """

    print("------------------------- Entered use_browser -------------------------")
    print("Query:", query)

    cmd = ["python", "browser_agent.py", "--playwright"]
    
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
    Edit a file using Vim based strictly on its current contents and explicit instructions.

    The tool has no external knowledge and must not guess or invent values.
    All edits must be fully and unambiguously specified in `query`
    (e.g. replace X with Y, delete a specific line, uncomment an exact line).

    Examples of valid instructions:
      - "Replace 'foo' with 'bar' in the config array."
      - "Uncomment the line 'overwrite.cli.url'."
      - "Insert the text ... above the closing bracket."

    Examples of invalid instructions:
      - "Fix the incorrect settings."
      - "Set this to the right value."    

    Returns a short report of applied changes or an explanation.
    """


    print("------------------------- Entered use_vim -------------------------")
    print("filename:", filename)
    print("Query:", query)


    human_delay_for_vim()

    session = get_session()
    try:
        session.start_vim(filename=filename)
        file_content = session.print_file_vim()
        result = run_file_edit_agent(
            query=query,
            file_content=file_content,
            big_file=len(file_content) > AgentConfig.READ_FILE_MAX_CHARS,
        )
        updated_file = result["updated_file"]
        explanation = result["explanation"]
        session.overwrite_vim(updated_file)
        session.end_vim()
        return explanation
    except Exception as e:
        # Recover the shell if Vim state becomes inconsistent mid-edit.
        try:
            session._vim_escape_hatch()
        except Exception:
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
    model=AgentConfig.SUMMARY_MODEL_NAME,  # cheap summarizer
    temperature=AgentConfig.SUMMARY_MODEL_TEMPERATURE,
)

model = ChatOpenAI(model=AgentConfig.MAIN_MODEL_NAME, temperature=AgentConfig.MAIN_MODEL_TEMPERATURE).bind_tools(tools=tools)
    #tool_choice={"type": "function", "function": {"name": "it_is_enough"}}  # <-- correct shape


# ---------- Nodes ----------
def decision_node(state: AgentState) -> AgentState:
    """
    Construct the next model call from the current trajectory state.

    This node refreshes the running summary when needed, rebuilds a bounded
    prompt, and returns the next assistant message plus updated state fields.
    """

    decision_steps = state.get("decision_steps", 0) + 1
    print("Step:", decision_steps)

    summary_updates = maybe_summarize_history(state, AgentConfig.SUMMARY_THRESHOLD)
    history_summary = summary_updates.get("history_summary", state.get("history_summary"))
    summarized_upto = summary_updates.get("summarized_upto", state.get("summarized_upto", 0))

    system_prompt_parts = [
        # These are stable environment assumptions the agent should treat as fixed.
        "You are a Linux web administrator operating on an Ubuntu 24.04 server.\n"
        "You have full root access (sudo -i).\n"
        "The system runs a LAMP stack (Linux, Apache, MariaDB, PHP) with Nextcloud (PHP 8.3.6).\n"
        "Standard system utilities are available, including curl and net-tools.\n",
        "A backup of the Nextcloud instance is available at /var/backups.\n\n"
    ]

    if AgentConfig.ENABLE_IN_CONTEXT_EXAMPLES:
        system_prompt_parts.extend([
            "Reference material:\n\n",
            f"Examples (Admin Routines):\n{examples_content}\n\n",
            f"Cheat Sheet (Nextcloud occ):\n{cheatsheet_content}\n\n",
        ])

    system_prompt_parts.extend([
        "Available tools:\n"
        "1. next_command(cmd: str) — execute one non-interactive shell command.\n"
        "2. use_browser(query: str) — interact with the Nextcloud web UI (prefer over curl).\n"
        "3. use_vim(filename: str, query: str) — edit files with explicit instructions.\n"
        "4. read_file(filename: str) — read file contents (large files are truncated).\n\n",

        "Behavior rules:\n"
        "- next_command must execute exactly one command. Do not chain multiple commands.\n"
        "- There is no interactive user; do not ask questions.\n"
        "- Act autonomously and verify changes when appropriate.\n"
        "- When the task is complete, call terminate with a brief summary and current status.\n"
        "- Regularly verify whether the original problem still persists.\n"
        "- If a verification shows that the problem is resolved, IMMEDIATELY call terminate().\n"
        "- Do not repeat the same diagnostic command (same file and same intent) unless you changed something relevant.\n"
        "- NEVER perform large or recursive restore operations (e.g. cp/rsync of whole directories like /var/backups or /var/www);"
    ])

    system_prompt = "".join(system_prompt_parts)

    # The prompt uses a short working window plus summary instead of full history.
    chat_history = build_model_messages(state["messages"], max_history=AgentConfig.MAX_HISTORY_WINDOW)

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

    response = invoke_with_retry(model, model_messages)


    return {
        "messages": [response],
        "history_summary": history_summary,
        "summarized_upto": summarized_upto,
        "decision_steps": decision_steps,
    }


# ---------- Build graph ----------
graph = StateGraph(AgentState)
graph.add_node("decision_node", decision_node)
tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)
graph.add_edge(START, "decision_node")
    
def route_decision(state: AgentState) -> str:
    """Route to tool execution when the last assistant message contains tool calls."""
    last = state["messages"][-1]
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
    """
    Decide whether tool execution should terminate the graph or continue.

    A `terminate` tool result is treated as the explicit stopping signal.
    """
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

    def handle_sigint(signum, frame):
        """Clean up the shared session before exiting on Ctrl+C."""
        print("\n[signal] SIGINT received, cleaning up...")
        cleanup_session()
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_sigint)

    result = None
    get_session()

    try:
        result = app.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=AgentConfig.problem_prompt
                    )
                ],
                "history_summary": None,
                "summarized_upto": 0,
                "decision_steps": 0,
            },
            config={"recursion_limit": AgentConfig.RECURSION_LIMIT},
        )

    finally:

        # ---------- Final output ----------
        print("Output:")
        if result is not None:
            for message in result["messages"]:
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()
        else:
            print("[NO RESULT — the agent crashed before producing output]")

        print(f"decision_node executions: {result.get('decision_steps', 0)}")

        # ---------- Cleanup ----------
        cleanup_session()
