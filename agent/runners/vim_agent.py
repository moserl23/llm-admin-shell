"""LangGraph-based file editing agent.

The agent inspects a numbered in-memory file snapshot, proposes minimal line-based
edits through tool calls, and returns the updated file plus a short explanation.
"""

# Langgraph
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

# Typing
from typing import TypedDict, Optional, Annotated, Sequence, List, Literal
from pydantic import BaseModel
import re

# Config
#from config import API_KEY
from dotenv import load_dotenv
load_dotenv()

# ---------- Global ----------
FILE_CACHE: Optional[str] = None


# ---------- Hyperparameters / Config ----------
class EditAgentConfig:
    """Static configuration for the edit agent runtime."""
    MODEL_NAME = "gpt-4.1"
    TEMPERATURE = 0.3
    RECURSION_LIMIT = 30


class AgentState(TypedDict):
    """Graph state containing the conversation history and file-size mode."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    file_size_mode: Literal["small", "big"]

# ---------- Tools ----------

class Edit(BaseModel):
    """Single line-based edit operation applied to the cached file."""
    op: Literal["replace", "insert_after", "insert_before", "delete"]
    start_line: int
    end_line: Optional[int] = None
    content: Optional[List[str]] = None

class Patch(BaseModel):
    """Collection of edit operations produced by the model."""
    edits: List[Edit]

@tool
def finalize_patch(patch: Patch, explanation: str) -> str:
    """
    Submit the final patch and explanation.
    """

    # DEBUG
    print("patch:", patch)
    print("explaination:", explanation)


    global FILE_CACHE
    if FILE_CACHE is None:
        raise ValueError("FILE_CACHE is empty; no file loaded.")

    # Strip synthetic line numbers before applying edits.
    numbered_lines = FILE_CACHE.splitlines()
    content_lines: List[str] = []
    for line in numbered_lines:
        parts = line.split(":", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            content_lines.append(parts[1].lstrip(" "))
        else:
            content_lines.append(line)

    # Apply edits from bottom to top so earlier line numbers are not shifted.
    edits_sorted = sorted(
        patch.edits,
        key=lambda e: (e.start_line, e.end_line or e.start_line),
        reverse=True,
    )

    for edit in edits_sorted:
        op = edit.op
        start_idx = edit.start_line - 1
        end_idx = (edit.end_line or edit.start_line) - 1
        new_content = edit.content or []

        if op == "replace":
            content_lines[start_idx:end_idx + 1] = new_content

        elif op == "delete":
            del content_lines[start_idx:end_idx + 1]

        elif op == "insert_before":
            content_lines[start_idx:start_idx] = new_content

        elif op == "insert_after":
            insert_at = end_idx + 1
            content_lines[insert_at:insert_at] = new_content

        else:
            raise ValueError(f"Unknown edit op: {op}")

    # Rebuild the numbered cache consumed by the inspection tools.
    FILE_CACHE = "\n".join(
        f"{i+1}: {line}"
        for i, line in enumerate(content_lines)
    )

    return explanation


@tool
def read_file_slice(
    start_line: int,
    num_lines: int = 20,
) -> str:
    """
    Return up to `num_lines` lines starting from `start_line` (1-based, inclusive).
    Used to inspect a specific region of the file.
    """
    global FILE_CACHE
    if FILE_CACHE is None:
        return "No file loaded."

    lines = FILE_CACHE.splitlines()
    total_lines = len(lines)

    start_idx = start_line - 1
    if start_idx >= total_lines:
        return f"Start line {start_line} is beyond end of file."

    end_idx = min(total_lines, start_idx + num_lines)
    sliced = lines[start_idx:end_idx]

    if not sliced:
        return f"No lines starting at {start_line}."

    return "\n".join(sliced)


@tool
def search_regex_window(
    pattern: str,
    before: int = 5,
    after: int = 5,
    max_matches: int = 20,
) -> str:
    """
    Search for a regex pattern in the file and return up to `max_matches` matches
    with `before` and `after` context lines for each match.
    """
    global FILE_CACHE
    if FILE_CACHE is None:
        return "No file loaded."

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    lines = FILE_CACHE.splitlines()
    results = []
    match_count = 0

    for i, line in enumerate(lines):
        if regex.search(line):
            match_count += 1
            if match_count > max_matches:
                break

            start = max(0, i - before)
            end = min(len(lines), i + after + 1)
            window_lines = lines[start:end]

            block = (
                f"Match {match_count}:\n"
                + "\n".join(window_lines)
            )
            results.append(block)

    if not results:
        return f"No matches found for regex '{pattern}'."

    return "\n\n".join(results)




@tool
def search_text_window(
    query: str,
    before: int = 5,
    after: int = 5,
    max_matches: int = 20,
) -> str:
    """
    Search for a text query in the file and return up to `max_matches` matches
    with `before` and `after` context lines for each match.
    """
    global FILE_CACHE
    if FILE_CACHE is None:
        return "No file loaded."

    lines = FILE_CACHE.splitlines()
    results = []
    match_count = 0

    for i, line in enumerate(lines):
        if query in line:
            match_count += 1
            if match_count > max_matches:
                break

            start = max(0, i - before)
            end = min(len(lines), i + after + 1)

            window_lines = lines[start:end]

            block = (
                f"Match {match_count}:\n"
                + "\n".join(window_lines)
            )
            results.append(block)

    if not results:
        return f"No matches found for '{query}'."

    return "\n\n".join(results)



@tool
def read_file() -> str:
    """
    Read the entire file content with numbered lines.
    """
    global FILE_CACHE
    if FILE_CACHE is None:
        return ""
    return FILE_CACHE


tools_small_file = [read_file, finalize_patch]
tools_big_file = [read_file_slice, search_regex_window, search_text_window, finalize_patch]
all_tools = tools_big_file + tools_small_file

# ---------- LLM client ----------
base_model = ChatOpenAI(model=EditAgentConfig.MODEL_NAME, temperature=EditAgentConfig.TEMPERATURE)


# ---------- Nodes ----------
def decision_node(state: AgentState) -> AgentState:
    """
    Run the model with the toolset appropriate for the current file size.

    The prompt emphasizes minimal inspection and requires the model to finalize
    through `finalize_patch`, even for read-only or no-op outcomes.
    """

    system_prompt = (
        "You are a file-editing agent. Your ONLY task is to apply the user's request to "
        "THIS file. You must ONLY make changes if the request requires them AND the file "
        "content confirms that the conditions are met.\n\n"

        "GENERAL RULES:\n"
        "- The file content is the single source of truth.\n"
        "- Use as FEW inspection tool calls as possible (ideal: 1, max: 3).\n"
        "- As soon as ANY tool returns the relevant line(s) with line numbers, STOP inspecting.\n"
        "- Never re-read a line already seen in any tool output.\n"
        "- Never add logic, templates, comments, or new structure. Only edit existing lines.\n"
        "- For read-only or summary requests, DO NOT modify the file (Patch(edits=[])).\n"
        "- - Never invent passwords, credentials, or configuration values unless the exact values are explicitly provided.\n\n"

        "=== EXAMPLE 1 — SIMPLE EDIT ===\n"
        "User request: 'Change port 5432 to 6432'.\n"
        "Tool result (search_text_window('5432')):\n"
        "  75: engine = \"postgresql\"\n"
        "  76: host   = \"127.0.0.1\"\n"
        "  77: port   = 5432\n"
        "  78: user   = \"demo\"\n\n"
        "This is enough information. CORRECT BEHAVIOR:\n"
        "→ STOP immediately.\n"
        "→ Create a patch replacing ONLY line 77 with 'port   = 6432'.\n"
        "→ NO more search or slice calls.\n\n"

        "=== EXAMPLE 2 — CONDITIONAL EDIT ===\n"
        "User request: 'If environment is \"staging\", change timezone to Europe/Paris'.\n"
        "Tool result (search_text_window('environment')):\n"
        "  5: name        = \"MyService\"\n"
        "  6: environment = \"production\"\n"
        "  7: timezone    = \"Europe/Vienna\"\n\n"
        "Condition is FALSE (environment ≠ \"staging\"). CORRECT BEHAVIOR:\n"
        "→ STOP immediately.\n"
        "→ NO edits.\n"
        "→ finalize_patch with Patch(edits=[]) and a short explanation.\n\n"

        "=== EXAMPLE 3 — SUMMARY REQUEST (READ-ONLY) ===\n"
        "User request: 'Summarize the database configuration'.\n"
        "Tool result (search_text_window('postgresql')):\n"
        "  75: engine = \"postgresql\"\n"
        "  76: host   = \"127.0.0.1\"\n"
        "  77: port   = 5432\n"
        "  78: user   = \"myservice\"\n\n"
        "CORRECT BEHAVIOR:\n"
        "→ STOP immediately.\n"
        "→ NO edits (Patch(edits=[])).\n"
        "→ Summary goes into explanation.\n\n"

        "=== EXAMPLE 4 — MULTI-LINE EDIT ===\n"
        "User request: 'Change host to 0.0.0.0 and port to 6543'.\n"
        "Tool result (search_text_window('postgresql')):\n"
        "  75: engine = \"postgresql\"\n"
        "  76: host   = \"127.0.0.1\"\n"
        "  77: port   = 5432\n\n"
        "CORRECT BEHAVIOR:\n"
        "→ STOP immediately.\n"
        "→ Patch with TWO replaces (line 76 and line 77).\n"
        "→ NO further inspection.\n\n"

        "PATCH RULES:\n"
        "- Use the Edit schema (op, start_line, end_line if needed, content if needed).\n"
        "- Minimal edits only. Do not repeat unchanged lines.\n"
        "- If no change is required or allowed, use Patch(edits=[]).\n"

        "FINALIZATION:\n"
        "Always finalize with: finalize_patch(patch=Patch(...), explanation=...).\n"
        "explanation must be a short plain-text sentence describing what changed or, "
        "for summary requests, summarizing the relevant lines."
    )

    # Small files can be read in one shot; larger files force targeted inspection.
    if state["file_size_mode"] == "small":
        allowed_tools = tools_small_file
    else:
        allowed_tools = tools_big_file

    model = base_model.bind_tools(allowed_tools)

    response = model.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response]}



# ---------- Build graph ----------
graph = StateGraph(AgentState)
graph.add_node("decision_node", decision_node)
tool_node = ToolNode(tools=all_tools)
graph.add_node("tool_node", tool_node)
graph.add_edge(START, "decision_node")
    
def route_decision(state: AgentState) -> str:
    """
    Route model outputs either to tool execution or back to the model loop.

    The graph stays in the decision loop until the model emits a tool call.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "__tool__"
    return "__decision__"

graph.add_conditional_edges(
    "decision_node",
    route_decision,
    path_map={
        "__tool__": "tool_node",
        "__decision__": "decision_node",
    },
)

def route_after_tool(state: AgentState) -> str:
    """
    Decide whether tool execution should terminate the graph or continue reasoning.

    `finalize_patch` is treated as the terminal tool because it both applies edits
    and packages the final explanation.
    """
    last = state["messages"][-1]

    if isinstance(last, ToolMessage) and last.name == "finalize_patch":
        return "__end__"

    return "__decision__"

graph.add_conditional_edges(
    "tool_node",
    route_after_tool,
    path_map={
        "__end__": END,
        "__decision__": "decision_node",
    },
)

# ---------- Compile & run ----------
app = graph.compile()

def run_file_edit_agent(
    query: str,
    file_content: str,
    big_file: bool,
) -> dict:
    """
    Execute the editing graph against a single file snapshot.

    The file is exposed to the agent as numbered text to stabilize line-based edits.
    Returns the updated unnumbered content together with the model's explanation.
    """

    # The agent edits by explicit line number, so the cache is always numbered.
    numbered_content = "\n".join(
        f"{i+1}: {line}"
        for i, line in enumerate(file_content.splitlines())
    )


    global FILE_CACHE
    FILE_CACHE = numbered_content

    file_size_mode = "big" if big_file else "small"

    result = app.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "file_size_mode": file_size_mode,
        },
        config={"recursion_limit": EditAgentConfig.RECURSION_LIMIT},
    )


    # The final explanation is emitted by the terminal `finalize_patch` tool call.
    explanation = None
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "finalize_patch":
            explanation = msg.content
    if explanation is None:
        explanation = "No explanation returned from finalize_patch."

    # Remove synthetic numbering before returning the updated file to the caller.
    unnumbered_lines = []
    for line in FILE_CACHE.splitlines():
        parts = line.split(":", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            unnumbered_lines.append(parts[1].lstrip())
        else:
            unnumbered_lines.append(line)
    unnumbered_content = "\n".join(unnumbered_lines)



    return {
        "updated_file": unnumbered_content,
        "explanation": explanation,
    }




if __name__ == "__main__":
    pass
    
    
        




