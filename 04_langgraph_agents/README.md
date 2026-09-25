# Module 04: LangGraph Agents & Tool Calling

This module shows how LangGraph achieves cyclical agent loops (ReAct) using `ToolNode` and conditional routing.

---

## 🔄 The Cyclical ReAct Loop in LangGraph

Unlike LangChain's older `AgentExecutor`, LangGraph models agent loops as an explicit graph:

```
                ┌─────────┐
                │  START  │
                └────┬────┘
                     │
                     ▼
             ┌───────────────┐
    ┌───────>│  Agent Node   │
    │        │  (Calls LLM)  │
    │        └───────┬───────┘
    │                │
    │         tools_condition
    │          ├── Has tool_calls? ──┐
    │          └── No tool calls     │
    │                  │             │
    │                  ▼             ▼
    │               ┌─────┐   ┌──────────────┐
    │               │ END │   │  ToolNode    │
    │               └─────┘   │(Runs Tools)  │
    │                         └──────┬───────┘
    │                                │
    └────────────────────────────────┘
           (Cycles back to Agent)
```

### Why this design is superior:
1. **Explicit Control**: You can insert validation nodes between `agent` and `tools` (e.g., security check or cost limiter).
2. **True Cycles**: The loop continues naturally until the LLM stops generating `tool_calls`.
3. **State Observability**: Every step records state history cleanly.

---

## 🧩 Key Components

- **`ToolNode`**: A prebuilt node that takes the last `AIMessage`, inspects its `tool_calls`, executes each tool, and outputs `ToolMessage` instances containing the observations.
- **`tools_condition`**: A built-in conditional router that checks whether the last message contains `tool_calls`. If yes, routes to `"tools"`; if no, routes to `END`.
- **`create_react_agent`**: A high-level factory function in `langgraph.prebuilt` that assembles this exact graph in a single line.

---

## 🚀 Running the Examples

```bash
python 04_langgraph_agents/01_langgraph_react_agent.py
python 04_langgraph_agents/02_prebuilt_create_react_agent.py
```
