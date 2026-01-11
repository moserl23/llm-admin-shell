# langgraph_agent.py
from typing_extensions import TypedDict
from typing import Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from config import API_KEY

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

import requests

# ----- Summarizing reducer -----
# (your snippet, with api_key wired in)
llm_summarizer = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=API_KEY)

def summarize_and_append(existing: List[BaseMessage], new: List[BaseMessage]):
    # keep last few messages as-is
    keep_recent = 2
    if len(existing) > keep_recent:
        summary = llm_summarizer.invoke(
            f"Summarize this chat so far:\n{existing[:-keep_recent]}"
        )
        return [summary] + existing[-keep_recent:] + new
    return existing + new

# ----- Tools (same as LangChain) -----
@tool("square_tool")
def square_tool(number: str) -> str:
    """Squares an integer given as a string."""
    n = int(number.strip())
    return str(n * n)

@tool("weather_tool")
def weather_tool(city: str) -> str:
    """Get quick current weather via wttr.in (one-line)."""
    url = f"http://wttr.in/{city.strip()}?format=3"
    resp = requests.get(url, timeout=8)
    resp.raise_for_status()
    return resp.text

tools = [square_tool, weather_tool]

# ----- State -----
class AgentState(TypedDict):
    # Use the summarizing reducer instead of add_messages
    messages: Annotated[List[BaseMessage], summarize_and_append]

# ----- Model bound to tools -----
llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=API_KEY)
llm_with_tools = llm.bind_tools(tools)

# ----- Nodes -----
def call_model(state: AgentState):
    """LLM step: take messages -> produce next assistant message (may include tool calls)."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)  # executes any tool calls found in the last assistant message

# ----- Routing -----
def route_after_model(state: AgentState):
    """If the last assistant message requested tools, go to tools; else finish."""
    last = state["messages"][-1]
    has_tool_calls = getattr(last, "tool_calls", None)
    return "tools" if has_tool_calls else END

# ----- Graph -----
graph = StateGraph(AgentState)
graph.add_node("llm", call_model)
graph.add_node("tools", tool_node)

graph.set_entry_point("llm")
graph.add_conditional_edges("llm", route_after_model, {"tools": "tools", END: END})
graph.add_edge("tools", "llm")  # after tools run, go back to the LLM

# Optional: persistence during multi-step runs
from langgraph.checkpoint.memory import MemorySaver
app = graph.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    out1 = app.invoke(
        {"messages": [HumanMessage(content="Look the weather for London, Paris, Vienna, New-York, Tokio, Santiago de Chile. Always tell me jokes between the cities.")]},
        config={"configurable": {"thread_id": "demo-1"}},   # required with a checkpointer
    )
    for m in out1["messages"]:
        role = m.type
        print(f"{role.upper()}: {getattr(m, 'content', '')}")
