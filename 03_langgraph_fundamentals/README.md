# Module 03: LangGraph Fundamentals

This module explains the core concepts and mechanics of **LangGraph**, the state-machine orchestration engine built by the LangChain team.

---

## 🧭 Why LangGraph? (Chains vs Graphs)

Traditional LLM chains (like LCEL or `LLMChain`) are **DAGs (Directed Acyclic Graphs)**. They flow in one direction:
```
Input ──> Step A ──> Step B ──> Step C ──> Output
```

However, real-world agentic systems require:
- **Loops & Cycles**: An agent might need to try an action, observe the error, correct itself, and try again until it succeeds.
- **Shared State**: An evolving state dictionary accessible across all steps.
- **Persistence & Checkpointing**: The ability to pause execution, inspect state, and resume later.
- **Human-in-the-Loop**: Halting execution to wait for a human decision before proceeding.

LangGraph implements these as a **State Machine**.

---

## 🏗️ Core Concepts of LangGraph

```
           ┌─────────┐
           │  START  │
           └────┬────┘
                │
                ▼
           ┌─────────┐
           │ Node A  │ (Takes State -> Returns State Update)
           └────┬────┘
                │
         Conditional Edge
           (Decision?)
          ┌─────┴─────┐
          ▼           ▼
     ┌─────────┐ ┌─────────┐
     │ Node B  │ │ Node C  │
     └────┬────┘ └────┬────┘
          │           │
          └─────┬─────┘
                ▼
            ┌───────┐
            │  END  │
            └───────┘
```

1. **State**:
   - A Python `TypedDict` or Pydantic class representing the schema of the graph's memory.
   - Every node receives this state and returns updates to it.

2. **Nodes**:
   - Python functions: `def my_node(state: MyState) -> dict:`
   - Returns a partial dictionary updating one or more keys in the state.

3. **Edges**:
   - **Fixed Edges** (`graph.add_edge("node_a", "node_b")`): Direct transition from node A to node B.
   - **Conditional Edges** (`graph.add_conditional_edges("node_a", routing_fn, path_map)`): Evaluates state at runtime to determine the next node.
   - **`START` and `END`**: Built-in entrypoint and exit terminals.

4. **Reducers (`operator.add` / `add_messages`)**:
   - By default, returning a key in a node *overwrites* the previous value.
   - Reducers tell LangGraph how to combine state updates, e.g., appending items to a list instead of overwriting.

---

## 🚀 Running the Examples

```bash
python 03_langgraph_fundamentals/01_basic_stategraph.py
python 03_langgraph_fundamentals/02_conditional_routing.py
python 03_langgraph_fundamentals/03_state_reducers.py
```
