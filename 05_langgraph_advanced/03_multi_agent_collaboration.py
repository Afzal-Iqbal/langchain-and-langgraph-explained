"""
==============================================================================
03_multi_agent_collaboration.py - Multi-Agent Supervisor Pattern in LangGraph
==============================================================================

This script demonstrates:
1. Designing a Multi-Agent system with specialized agent nodes.
2. A Supervisor Node that acts as a router/coordinator.
3. State handoff between Researcher Agent, Coder Agent, and Supervisor.
4. Concluding the graph once all sub-tasks are solved.
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. State Definition for Collaboration
# ---------------------------------------------------------------------------
class MultiAgentState(TypedDict):
    task: str
    research_notes: str
    code_output: str
    review_status: str
    next_step: str


# ---------------------------------------------------------------------------
# 2. Specialized Agent Nodes
# ---------------------------------------------------------------------------
def supervisor_node(state: MultiAgentState) -> dict:
    """Evaluates task progress and decides which specialist to activate next."""
    print("  [Supervisor] Assessing current project state...")

    # If no research done, assign to researcher
    if not state.get("research_notes"):
        print("  [Supervisor Decision] -> Routing to RESEARCHER")
        return {"next_step": "researcher"}

    # If research is complete but no code produced, assign to coder
    if not state.get("code_output"):
        print("  [Supervisor Decision] -> Routing to CODER")
        return {"next_step": "coder"}

    # Both complete! Mark finished
    print("  [Supervisor Decision] -> All requirements met! Finishing workflow.")
    return {"next_step": "FINISH", "review_status": "APPROVED"}


def researcher_node(state: MultiAgentState) -> dict:
    """Specialist agent that conducts research and drafts technical requirements."""
    print("  [Researcher Agent] Analyzing prompt and gathering technical specifications...")
    notes = (
        "Specification: Fibonacci Generator function in Python.\n"
        "Requirements: Must handle edge cases (n <= 0), use memoization/generator for O(1) space."
    )
    return {"research_notes": notes}


def coder_node(state: MultiAgentState) -> dict:
    """Specialist agent that writes verified code according to research notes."""
    print("  [Coder Agent] Writing Python implementation from specifications...")
    code = (
        "def fibonacci(n: int):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        yield a\n"
        "        a, b = b, a + b\n"
    )
    return {"code_output": code}


# ---------------------------------------------------------------------------
# 3. Router Function for Supervisor
# ---------------------------------------------------------------------------
def route_next_agent(state: MultiAgentState) -> Literal["researcher", "coder", "__end__"]:
    next_step = state.get("next_step")
    if next_step == "researcher":
        return "researcher"
    elif next_step == "coder":
        return "coder"
    else:
        return "__end__"


# ---------------------------------------------------------------------------
# 4. Construct Multi-Agent StateGraph
# ---------------------------------------------------------------------------
def build_multi_agent_system():
    builder = StateGraph(MultiAgentState)

    # Register nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)

    # Flow starts at Supervisor
    builder.add_edge(START, "supervisor")

    # Supervisor conditional routing
    builder.add_conditional_edges(
        "supervisor",
        route_next_agent,
        {
            "researcher": "researcher",
            "coder": "coder",
            "__end__": END,
        }
    )

    # Specialists report back to Supervisor when their sub-task completes
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("coder", "supervisor")

    return builder.compile()


def main():
    print("--- Multi-Agent Collaboration Workflow (Supervisor Pattern) ---\n")
    app = build_multi_agent_system()

    initial_input = {
        "task": "Build an optimized Fibonacci sequence generator in Python",
        "research_notes": "",
        "code_output": "",
        "review_status": "PENDING",
        "next_step": "",
    }

    print(f"Goal: {initial_input['task']}\n")

    final_state = app.invoke(initial_input)

    print("\n================ FINAL DELIVERABLE ================")
    print(f"Status: {final_state['review_status']}")
    print("\nResearch Notes:\n" + final_state["research_notes"])
    print("\nGenerated Code:\n" + final_state["code_output"])


if __name__ == "__main__":
    main()
