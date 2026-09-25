# Module 05: Advanced LangGraph (Persistence, Human-in-the-Loop & Multi-Agent)

This module dives into enterprise-grade features that set LangGraph apart from any traditional chaining library:

1. **State Persistence & Checkpoints**
2. **Human-in-the-Loop (Approval Breakpoints)**
3. **Multi-Agent Orchestration (Supervisor Pattern)**

---

## 💾 1. Persistence & Checkpoints

LangGraph checkpointers (`MemorySaver`, `SqliteSaver`, `PostgresSaver`) save a snapshot of the graph's state after every node executes.

```
Node A ──> [Checkpoint 1 saved] ──> Node B ──> [Checkpoint 2 saved] ──> Node C
```

### Why this is critical:
- **Thread Isolation**: Multiple users have separate conversation threads (`thread_id`).
- **Resilience**: If a server crashes midway, the graph resumes from the last checkpoint without rerunning previous steps.
- **Time Travel**: You can inspect past state snapshots, rewind, and re-execute from an earlier point.

---

## ⏸️ 2. Human-in-the-Loop (Breakpoints)

For sensitive operations (e.g., sending emails, refunding payments, executing code, modifying databases), an agent should not act completely autonomously.

```
Agent Node (wants to delete user record)
               │
               ▼
       [ BREAKPOINT ⏸️ ]  --> Execution paused. Snapshot saved.
               │
      Human inspects state & approves / modifies / rejects
               │
               ▼
         Graph Resumes
```

LangGraph provides native support via `interrupt_before=["tool_node"]` or the dynamic `interrupt()` function.

---

## 👥 3. Multi-Agent Systems (Supervisor Pattern)

Complex tasks are best solved by specialized agents rather than a single generalist prompt:

```
                      ┌──────────────────────┐
                      │      Supervisor      │
                      │  (Planner & Router)  │
                      └──────────┬───────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       ┌───────────────────┐           ┌───────────────────┐
       │  Research Agent   │           │    Coder Agent    │
       │(Gathers technical)│           │ (Writes & verifies│
       │   information     │           │       code)       │
       └───────────────────┘           └───────────────────┘
```

The Supervisor inspects the current progress, assigns the next turn to the appropriate specialist, and terminates once the user's objective is met.

---

## 🚀 Running the Examples

```bash
python 05_langgraph_advanced/01_persistence_and_checkpoints.py
python 05_langgraph_advanced/02_human_in_the_loop.py
python 05_langgraph_advanced/03_multi_agent_collaboration.py
```
