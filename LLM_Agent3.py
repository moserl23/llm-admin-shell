# Langgraph
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Typing
from typing import TypedDict, Optional, Annotated, Sequence

# Config
from config import API_KEY
from utils import examples_content, cheatsheet_content


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ---------- Nodes ----------
def decision_node(state: AgentState) -> AgentState:
    system_prompt_parts = [
        "Reference:\n",
        f"Examples (Admin Routines):\n{examples_content}\n\n",
        f"Cheat Sheet (occ):\n{cheatsheet_content}\n\n",
        "You are an ops remediator for Ubuntu 24.04 (LAMP, Nextcloud PHP 8.3.6).\n\n",
        "=== Guidelines ===\n",
        "- Prefer read-only checks first; avoid destructive ops.\n"
        "- One command per call; keep it ONE line.\n"
        "- Keep output short; use grep/tail/head for large logs.\n"
        "- No interactive/full-screen tools.\n"
        "- You are root via SSH (sudo -i).\n"
        "  export SYSTEMD_URLIFY=0\n"
        "  export SYSTEMD_PAGER=\n"
        "  export SYSTEMD_COLORS=0\n\n",
        "=== Nextcloud ===\n",
        "- Install path: /var/www/nextcloud\n"
        "- Run occ as: /usr/bin/php /var/www/nextcloud/occ\n\n",
        "=== Logs ===\n",
        "- Apache error: /var/log/apache2/nextcloud.local-error.log\n"
        "- Apache access: /var/log/apache2/nextcloud.local-access.log\n\n",
        "=== System tools ===\n",
        "- curl (HTTP), net-tools, top (batch: top -b -n 1)\n\n",
        "=== Agent tools ===\n",
        "1. `next_command(cmd: str)` — run a shell command.\n"
        "2. `use_browser(query: str)` — interact with a web UI.\n"
        "3. `use_vim(filename: str, query: str)` — edit files.\n",
    ]
    system_prompt = "".join(system_prompt_parts)
    response = model.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response]}


# ---------- Tools ----------
    
@tool
def next_command(cmd: str) -> str:
    """
    Executes a given shell command on the server in an interactive terminal session.

    Args:
        cmd (str): The full command string to be executed in the shell (e.g., 'ls -la' or 'python script.py').

    Returns:
        str: The standard output or result of running the command.
    """
    # I will write this code later!
    print("cmd", cmd)
    return "Success"

@tool
def use_browser(query: str) -> str:
    """
    Performs actions on a web interface through an automated browser session.

    Args:
        query (str): A natural language description of what to do in the browser.
            Examples include logging into a website, navigating inside the web-application,
            clicking buttons, or filling out forms.

    Returns:
        str: A short summary or result of the performed browser actions.

    Notes:
        - Use this tool whenever interaction with a website or web UI is required.
        - The query should describe the desired goal or action sequence clearly.
    """
    print("browser-query:", query)
    return "Success"


@tool
def use_vim(filename: str, query: str) -> str:
    """
    Edits or modifies the specified file in a text editor (simulating Vim behavior) based on the given instructions.

    Args:
        filename (str): The name or path of the file to open or edit.
        query (str): A natural language description of the desired file modifications.
            Examples include adding new lines, changing specific text, deleting sections,
            or saving the file after editing.

    Returns:
        str: A short summary or confirmation of the changes performed on the file.

    Notes:
        - Use this tool whenever you need to open, read, or modify a file's contents.
        - The query should clearly describe what to edit or update in the file.
    """
    print("vim-filename:", filename)
    print("vim-query:", query)
    return "Success"



tools = [next_command, use_browser, use_vim]
    
# ---------- LLM client ----------
model = ChatOpenAI(model="gpt-4o", api_key=API_KEY).bind_tools(tools=tools)
    #tool_choice={"type": "function", "function": {"name": "it_is_enough"}}  # <-- correct shape


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
    return "__end__"

graph.add_conditional_edges(
    "decision_node",
    route_decision,
    path_map={
        "__tool__": "tool_node",
        "__end__": END,  # special label to end the graph
    },
)

# loop back
graph.add_edge("tool_node", "decision_node")

# ---------- Compile & run ----------
app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({"messages": [HumanMessage(content="Nextcloud is responding with 500 Error! Start with some browser error shotting and vim file editing!")]})

    print("Output:")
    for message in result["messages"]:
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

    # Add a small message