"""
==============================================================================
01_basic_stategraph.py - Minimal Working StateGraph in LangGraph
==============================================================================

This script demonstrates:
1. Defining graph state using `TypedDict`.
2. Writing state-transforming node functions.
3. Connecting nodes sequentially with `add_edge`.
4. Using `START` and `END` nodes.
5. Compiling and running the graph.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. Define the State Schema
# ---------------------------------------------------------------------------
class PipelineState(TypedDict):
    input_text: str
    cleaned_text: str
    word_count: int
    summary: str


# ---------------------------------------------------------------------------
# 2. Define Node Functions
# ---------------------------------------------------------------------------
def clean_text_node(state: PipelineState) -> dict:
    """Cleans up raw input whitespace and lowercases string."""
    print("  -> Executing: clean_text_node")
    cleaned = state["input_text"].strip().lower()
    return {"cleaned_text": cleaned}


def count_words_node(state: PipelineState) -> dict:
    """Counts total words from the cleaned text."""
    print("  -> Executing: count_words_node")
    words = state["cleaned_text"].split()
    return {"word_count": len(words)}


def summarize_node(state: PipelineState) -> dict:
    """Creates a final summary message combining accumulated state."""
    print("  -> Executing: summarize_node")
    summary = f"Processed text with {state['word_count']} words: '{state['cleaned_text']}'"
    return {"summary": summary}


# ---------------------------------------------------------------------------
# 3. Assemble and Compile the Graph
# ---------------------------------------------------------------------------
def build_graph():
    # Initialize builder with state schema
    builder = StateGraph(PipelineState)

    # Register nodes
    builder.add_node("clean_text", clean_text_node)
    builder.add_node("count_words", count_words_node)
    builder.add_node("summarize", summarize_node)

    # Wire up edges: START -> clean_text -> count_words -> summarize -> END
    builder.add_edge(START, "clean_text")
    builder.add_edge("clean_text", "count_words")
    builder.add_edge("count_words", "summarize")
    builder.add_edge("summarize", END)

    # Compile the graph into an executable runnable
    return builder.compile()


def main():
    print("--- 1. Compiling Basic StateGraph ---")
    app = build_graph()

    initial_input = {
        "input_text": "   LangGraph enables Cyclic Workflows and First-Class State!   "
    }
    print(f"\nInitial State: {initial_input}\n")

    print("--- 2. Executing Graph (invoke) ---")
    final_state = app.invoke(initial_input)

    print("\n--- 3. Final State Results ---")
    for key, value in final_state.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
