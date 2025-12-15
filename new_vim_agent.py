# Langgraph
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

# Typing
from typing import TypedDict, Optional, Annotated, Sequence, List, Dict, Literal
from pydantic import BaseModel
import re

# Config
from config import API_KEY

# ---------- Global ----------
FILE_CACHE: Optional[str] = None


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    file_size_mode: Literal["small", "big"]

# ---------- Tools ----------

class Edit(BaseModel):
    op: Literal["replace", "insert_after", "insert_before", "delete"]
    start_line: int
    end_line: Optional[int] = None
    content: Optional[List[str]] = None

class Patch(BaseModel):
    edits: List[Edit]

@tool
def finalize_patch(patch: Patch, explanation: str) -> str:
    """
    Submit the final patch and explanation.
    """

    print("patch:", patch)
    print("explaination:", explanation)


    global FILE_CACHE
    if FILE_CACHE is None:
        raise ValueError("FILE_CACHE is empty; no file loaded.")

    # Split into lines and strip leading "<n>:" prefixes to get raw content
    numbered_lines = FILE_CACHE.splitlines()
    content_lines: List[str] = []
    for line in numbered_lines:
        # Expect format "N: rest of line"
        parts = line.split(":", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            content_lines.append(parts[1].lstrip(" "))
        else:
            # Fallback if line is not numbered as expected
            content_lines.append(line)

    # Apply edits from bottom to top so indices stay valid
    edits_sorted = sorted(
        patch.edits,
        key=lambda e: (e.start_line, e.end_line or e.start_line),
        reverse=True,
    )

    for edit in edits_sorted:
        op = edit.op
        start_idx = edit.start_line - 1  # 1-based -> 0-based
        end_idx = (edit.end_line or edit.start_line) - 1  # inclusive
        new_content = edit.content or []

        if op == "replace":
            # Replace lines [start_idx, end_idx] with new_content
            content_lines[start_idx:end_idx + 1] = new_content

        elif op == "delete":
            # Delete lines [start_idx, end_idx]
            del content_lines[start_idx:end_idx + 1]

        elif op == "insert_before":
            # Insert new_content before start_line
            content_lines[start_idx:start_idx] = new_content

        elif op == "insert_after":
            # Insert after the given region (end_line if provided, else start_line)
            insert_at = end_idx + 1
            content_lines[insert_at:insert_at] = new_content

        else:
            raise ValueError(f"Unknown edit op: {op}")

    # Re-number lines and write back into FILE_CACHE
    FILE_CACHE = "\n".join(
        f"{i+1}: {line}"
        for i, line in enumerate(content_lines)
    )

    # You can return the updated content or just "OK"
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

    start_idx = start_line - 1  # 0-based
    if start_idx >= total_lines:
        return f"Start line {start_line} is beyond end of file."

    end_idx = min(total_lines, start_idx + num_lines)  # exclusive
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
            end = min(len(lines), i + after + 1)  # end is exclusive
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
            end = min(len(lines), i + after + 1)  # end is exclusive

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
base_model = ChatOpenAI(model="gpt-4.1", api_key=API_KEY, temperature=0.3)


# ---------- Nodes ----------
def decision_node(state: AgentState) -> AgentState:

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


    '''
    system_prompt = (
        "You are a file-editing agent. Your ONLY task is to apply the user's request to "
        "THIS file, but ONLY if the requested conditions actually match the file content.\n\n"

        "GENERAL RULES:\n"
        "- File content is the single source of truth.\n"
        "- Use as FEW inspection tool calls as possible (ideal: 1, max: 3).\n"
        "- As soon as a tool output shows the exact line number and content you need, "
        "IMMEDIATELY stop inspecting and prepare the patch.\n"
        "- Never re-read any line that already appeared in previous tool output.\n"
        "- Never add comments, logic, conditions, templates, or structure to the file. "
        "Only edit the existing lines.\n\n"

        "EXAMPLE OF CORRECT BEHAVIOR:\n"
        "User request: 'Change port 5432 to 6432'.\n"
        "Tool result from search_text_window('5432'):\n"
        "  75: engine = \"postgresql\"\n"
        "  76: host   = \"127.0.0.1\"\n"
        "  77: port   = 5432\n"
        "  78: user   = \"demo\"\n\n"
        "This output already contains EVERYTHING needed:\n"
        "- The target line number: 77\n"
        "- The exact old value: 5432\n"
        "- The context around it\n\n"
        "CORRECT ACTION:\n"
        "→ Immediately call finalize_patch with a Patch that REPLACES line 77 only\n"
        "   (e.g., 'port   = 6432').\n"
        "→ Do NOT call read_file_slice.\n"
        "→ Do NOT perform any more inspections.\n\n"

        "PATCH RULES:\n"
        "- Use the Edit schema (op, start_line, end_line if needed, content if needed).\n"
        "- Minimal edits ONLY. Do not repeat unchanged lines.\n"
        "- If conditions do not match, or no change is needed, return Patch(edits=[]).\n\n"

        "FINALIZATION:\n"
        "Always finish with: finalize_patch(patch=Patch(...), explanation=...).\n"
        "The explanation must briefly describe why the change was applied or skipped."
    )    
    '''

    '''
    system_prompt = (
        "You are a file-editing agent. Your job is to apply the user's request to THIS file.\n\n"
        "TOOL USE:\n"
        "- Use as FEW inspection tools as possible (ideally 1, max 3).\n"
        "- If a tool result already shows the exact line you must edit (with its line number), "
        "you MUST NOT call any more inspection tools. Immediately prepare the patch.\n"
        "- Never re-read lines that already appeared in any previous tool output.\n\n"
        "PATCH:\n"
        "- Use the Edit schema (op, start_line, end_line if needed, content if needed).\n"
        "- Only change the requested values (minimal edits, no unchanged lines).\n"
        "- Always finish by calling finalize_patch(patch=Patch(...), explanation=<brief-explanation>). "
        "If nothing needs to be changed, use Patch(edits=[])."
    )
    '''
    
    '''
    system_prompt = (
        "You are a file-editing agent. Your ONLY task is to execute the user's request "
        "on THIS file, but ONLY if the conditions in the request are actually true in "
        "the file content.\n\n"

        "FILE CONTENT IS THE SINGLE SOURCE OF TRUTH.\n"
        "- If the user says 'If X then change Y', you MUST read X from the file using "
        "inspection tools and compare it literally.\n"
        "- If the condition is FALSE, you MUST make NO changes.\n"
        "- Never assume, infer, or hallucinate values. Only trust what appears in tool output.\n"
        "- Never insert logic, templates, conditions, comments, or new structure into the file.\n\n"

        "TOOL USAGE:\n"
        "- Use as FEW inspection tools as possible (ideal: 1–3 calls).\n"
        "- As soon as a tool reveals all required information (e.g., both the condition "
        "and the line to edit), STOP inspecting and prepare the patch immediately.\n"
        "- Never re-inspect lines you have already seen.\n\n"

        "PATCH RULES:\n"
        "- Produce a minimal edit patch using the Edit schema "
        "(op, start_line, end_line if needed, content if needed).\n"
        "- Do not include unchanged lines.\n"
        "- Line numbers must match exactly what you saw in tool output.\n"
        "- If the condition is FALSE, or no edit is needed, or you are uncertain, "
        "you MUST call finalize_patch with Patch(edits=[]).\n\n"

        "Always finish by calling finalize_patch(patch=Patch(...), explanation=...). "
        "The explanation must briefly state why the change was or was not applied."
    )
    '''

    '''
    system_prompt = (
        "You are a file-editing agent. Your job is to execute the user's edit request "
        "on THIS file, not to add new logic or conditionals.\n\n"
        "Interpret any 'if ... then ...' only as a condition to check the current file "
        "and decide whether to edit. Never write conditionals, templates, or code into the file.\n\n"
        "Use inspection tools only when needed, and use as FEW tool calls as possible "
        "(ideally 1–3). As soon as you find the relevant lines, stop inspecting and "
        "prepare the update list.\n\n"
        "Always finish by calling finalize_patch with a Patch object using the Edit schema "
        "(op, start_line, end_line if needed, content if needed). If no edits are required, call finalize_patch with an empty 'edits' list."
    )
    '''

    '''
    system_prompt = (
        "You are a file-editing agent and your task is to come up with an update list "
        "for the current file. You have tools available to inspect the file content.\n\n"
        "After inspection (max 3/4 inspection tools), call the finalize_patch tool using the Edit schema "
        "(op, start_line, end_line if needed, content if needed).\n\n"
        "If no changes are required or needed, or you didn't find anything useful, or you "
        "are uncertain, just call finalize_patch with an empty edits list."
        "If the conditions (like if) do not apply, or edits are not required, you MUST return an empty list for finalize_patch."
    )
    '''

    '''
    system_prompt = (
        "You are a file-editing agent.\n"
        "Don't guess file contents — use inspection tools whenever you need information.\n"
        "Do not call tools repeatedly for the same lines once you have already inspected them.\n"
        #"Only do direct file edits — if conditions are not met, don't apply changes.\n"
        "Do only a FEW tool calls!\n"
        "Do not ask or repeat questions.\n\n"
        "Finally, call `finalize_patch` with a Patch object:\n"
        "- Minimal edits only\n"
        "- No unchanged lines\n"
        "- Correct line numbers from tool output\n"
        "- Use the Edit schema (op, start_line, end_line if needed, content if needed)\n"
        "- If no changes are needed, you MUST call `finalize_patch` with an empty edits list.\n"
    )
    '''

    # Choose toolset based on file_size_mode
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
    last = state["messages"][-1]
    # If the LLM asked to call a tool, go to the ToolNode
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
    last = state["messages"][-1]

    # If the last message is the result of finalize_patch -> end
    if isinstance(last, ToolMessage) and last.name == "finalize_patch":
        return "__end__"

    # Otherwise, go back to the decision node
    return "__decision__"

graph.add_conditional_edges(
    "tool_node",
    route_after_tool,
    path_map={
        "__end__": END,
        "__decision__": "decision_node",
    },
)


def run_file_edit_agent(query: str, file_content: str) -> dict:


    # Add 1-based line numbers
    numbered_content = "\n".join(
        f"{i+1}: {line}"
        for i, line in enumerate(file_content.splitlines())
    )


    global FILE_CACHE
    FILE_CACHE = numbered_content

    result = app.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "file_size_mode": "big",
        },
        config={"recursion_limit": 20},
    )


    # Extract final tool message (which contains the explanation)
    explanation = None
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "finalize_patch":
            explanation = msg.content
    if explanation is None:
        explanation = "No explanation returned from finalize_patch."

    # get rid of numbering
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


# ---------- Compile & run ----------
app = graph.compile()

if __name__ == "__main__":

    # Read the file from the current directory
    with open("play_example_config.toml", "r", encoding="utf-8") as f:
        raw_content = f.read()

    # Add 1-based line numbers
    numbered_content = "\n".join(
        f"{i+1}: {line}"
        for i, line in enumerate(raw_content.splitlines())
    )

    # Store in global cache for tools
    FILE_CACHE = numbered_content

    # Ask the user for an editing query
    editing_query = input("Query: ")

    # Decide file size mode based on character length
    file_size_mode = "big" if len(raw_content) > 4000 else "small"

    # Invoke the graph
    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=editing_query)
            ],
            "file_size_mode": file_size_mode,
        },
        config={"recursion_limit": 20},
    )

    
    # Message History
    print("Output:")
    for message in result["messages"]:
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
    
    
        
    # Final File result
    #print("----------------------------- Updated File -----------------------------")
    #print(FILE_CACHE)




