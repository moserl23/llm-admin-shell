from typing import TypedDict

from langgraph.graph import StateGraph, END


class State(TypedDict):
    message: str


def greeter(state: State) -> State:
    state["message"] = "Hello " + state["message"]
    return state


def main() -> None:
    graph = StateGraph(State)
    graph.add_node("greeter", greeter)
    graph.set_entry_point("greeter")
    graph.add_edge("greeter", END)

    app = graph.compile()
    result = app.invoke({"message": "Lorenz"})
    print(result["message"])


if __name__ == "__main__":
    main()
