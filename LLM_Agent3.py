# Langgraph
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Typing
from typing import TypedDict, Optional, Annotated, Sequence, List
from pydantic import BaseModel

# Config
from config import API_KEY
from utils import examples_content, cheatsheet_content, ShellSession, init_env_and_log_offsets, read_new_logs
import subprocess

# additional tool code
from vim_tool import make_use_vim
from new_vim_agent import run_file_edit_agent

# global variables
global_session = ShellSession()
global_session.connect_root_setSentinel()
init_env_and_log_offsets(global_session)
# vim-llm
vim_planner = ChatOpenAI(model="gpt-4.1", temperature=0.3, api_key=API_KEY)

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
        "2. `use_browser(query: str)` — interact with the web UI (prefer over curl).\n"
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
    print("------------------------- Entered next_command -------------------------")
    print("cmd:", cmd)
    raw_result = global_session.run_cmd(cmd)
    return raw_result[-5000:] # Value is hard coded!



@tool
def use_browser(query: str) -> str:
    """
    Automate browser interactions on nextcloud.local.

    Args:
        query (str): A short natural-language instruction or command sequence for the browser tool.

    Returns:
        str: A brief summary of the browser tasks performed.
    """

    print("------------------------- Entered use_browser -------------------------")

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
    #print("\n\nBrowser Tool Output:")
    #print(stdout)
    #print("Browser Tool End\n\n")

    clean_output = "\n".join(
        line for line in stdout.splitlines() if not line.startswith("Query:")
    ).strip()


    return clean_output


@tool
def use_vim(filename: str, query: str) -> str:
    """
    Edit or modify a file using Vim-like keystrokes planned by an LLM.

    Args:
        filename (str): Full absolute path of the file to open or edit.
        query (str): Natural language description of the desired edits.

    Returns:
        str: Status message indicating success or failure of the edit.
    """

    print("vim-tool:")
    print("filename:", filename)
    print("query:", query)

    global_session.start_vim(filename=filename)

    file_content = global_session.print_file_vim()

    result = run_file_edit_agent(query=query, file_content=file_content)

    updated_file  = result["updated_file"]
    explanation   = result["explaination"]

    global_session.overwrite_vim(updated_file)

    global_session.end_vim()

    return explanation


tools = [next_command, use_browser, use_vim]
    
# ---------- LLM client ----------
model = ChatOpenAI(model="gpt-4o", api_key=API_KEY, temperature=0.1).bind_tools(tools=tools)
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
    result = app.invoke({"messages": [HumanMessage(content="In the root directory there is a file called ~/my_config.toml. Use the use_vim tool to perform an edit. Change in the server section the port to 7711.")]})

    print("Output:")
    for message in result["messages"]:
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


    read_new_logs(global_session)
    global_session.close()