# Module 02: LangChain Agents & Tools

This module explores how LangChain interfaces with external functions (tools) and orchestrates basic agent execution loops using `AgentExecutor`.

---

## 🛠️ What is an Agent?

Unlike a fixed chain that executes in a single deterministic sequence:
```
Prompt ──> LLM ──> Parser ──> Output
```

An **Agent** uses the LLM as a reasoning engine to decide:
1. **Which action/tool** to take (e.g., search web, query database, calculate math).
2. **What input** to pass to that tool.
3. **When to stop** and return the final answer.

```
                  ┌──────────────────────┐
                  │      User Input      │
                  └──────────┬───────────┘
                             │
                             ▼
                    ┌─────────────────┐
       ┌───────────>│    LLM Engine   │<───────────┐
       │            └────────┬────────┘            │
       │                     │                     │
       │             Needs tool call?              │
       │              ├── No  ──> Return Final     │
       │              └── Yes                      │
       │                     │                     │
       │                     ▼                     │
       │            ┌─────────────────┐            │
       │            │  Execute Tool   │            │
       │            └────────┬────────┘            │
       │                     │                     │
       └─────────────────────┴─────────────────────┘
                 Observation returned
```

---

## 🧰 Tools & The `@tool` Decorator

Tools are modular functions exposed to the model. LangChain reads:
- The Python function signature and type hints.
- The docstring (which becomes the tool description that instructs the LLM *when* and *how* to use it).
- Optional Pydantic argument schemas for strict type validation.

---

## ⚠️ Why `AgentExecutor` was Superseded by LangGraph

In earlier versions of LangChain, the `AgentExecutor` class handled agent loops. However:
- It was a "black box" that made modifying execution loops difficult.
- It lacked clean state persistence (e.g. paused workflows, rollback).
- Multi-agent coordination was difficult and fragile.
- Human-in-the-loop approvals were hard to implement reliably.

LangChain officially transitioned agent architectures to **LangGraph**. However, understanding tools and agents in LangChain provides the bridge to LangGraph.

---

## 🚀 Running the Examples

```bash
python 02_langchain_agents/01_custom_tools.py
python 02_langchain_agents/02_react_agent_executor.py
```
