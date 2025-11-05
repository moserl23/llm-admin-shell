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
from utils import examples_content, cheatsheet_content, ShellSession
import subprocess


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ---------- Nodes ----------
def decision_node(state: AgentState) -> AgentState:
    system_prompt_parts = [
        "Reference:\n",
        f"Examples (Admin Routines):\n{examples_content}\n\n",
        f"Cheat Sheet (occ):\n{cheatsheet_content}\n\n",
        "You are an ops remediator for Ubuntu 24.04 (LAMP, Nextcloud PHP 8.3.6).\n\n",
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
    Execute a single, non-interactive shell command on the Ubuntu 24.04 host (root).

    Args:
        cmd (str): The full command to run (exactly ONE line, e.g. "ls -la /var/log").

    Returns:
        str: The trimmed stdout/stderr result of the command.

    Usage guidelines:
        - Aoid destructive operations.
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
    # I will write this code later!
    print("cmd", cmd)
    return "Success"



@tool
def use_browser(query: str) -> str:
    """
    Automate browser interactions on nextcloud.local.

    Args:
        query (str): A short natural-language instruction or command sequence for the browser tool.

    Returns:
        str: A brief summary of the browser tasks performed.
    """
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

    stdout, stderr = process.communicate(query + "\n")  # Write the query

    if process.returncode != 0:
        return f"Browser tool error: {stderr.strip()}"

    print("Query:", query)
    print("\n\nBrowser Tool Output:")
    print(stdout)
    print("Browser Tool End\n\n")

    clean_output = "\n".join(
        line for line in stdout.splitlines() if not line.startswith("Query:")
    ).strip()

    print("Query:", query)
    print(clean_output)

    return clean_output



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
        - Use this tool whenever you need to modify a file's contents.
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
    result = app.invoke({"messages": [HumanMessage(content="Call the browser-tool to check the website.")]})

    print("Output:")
    for message in result["messages"]:
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

    # Add a small message