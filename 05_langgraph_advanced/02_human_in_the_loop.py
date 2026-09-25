"""
==============================================================================
02_human_in_the_loop.py - Human Approval Breakpoints in LangGraph
==============================================================================

This script demonstrates:
1. Compiling a graph with `interrupt_before` to pause before a sensitive node.
2. Checking the paused execution state (`state.next`).
3. Approving or editing the pending action.
4. Resuming the graph seamlessly with `app.invoke(None, config)`.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class TransferState(TypedDict):
    sender: str
    recipient: str
    amount: float
    status: str
    confirmation_code: str


def draft_transfer_node(state: TransferState) -> dict:
    """Prepares and validates the transfer request details."""
    print(f"  [draft_transfer_node] Drafting transfer of ${state['amount']} to {state['recipient']}...")
    return {
        "status": "Awaiting_Human_Approval"
    }


def execute_transfer_node(state: TransferState) -> dict:
    """Performs the actual critical financial transaction."""
    print(f"  [execute_transfer_node] EXECUTING TRANSFER: ${state['amount']} sent to {state['recipient']}!")
    return {
        "status": "Transferred_Successfully",
        "confirmation_code": f"TXN-{int(state['amount'] * 997)}"
    }


def build_approval_workflow():
    checkpointer = MemorySaver()
    builder = StateGraph(TransferState)

    builder.add_node("draft_transfer", draft_transfer_node)
    builder.add_node("execute_transfer", execute_transfer_node)

    builder.add_edge(START, "draft_transfer")
    builder.add_edge("draft_transfer", "execute_transfer")
    builder.add_edge("execute_transfer", END)

    # CRITICAL: Pause execution BEFORE 'execute_transfer' runs!
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute_transfer"]
    )


def main():
    print("--- Human-in-the-Loop Approval Demo ---")
    app = build_approval_workflow()

    config = {"configurable": {"thread_id": "transfer_task_404"}}

    initial_request = {
        "sender": "Alice",
        "recipient": "Bob Corp",
        "amount": 5000.00,
        "status": "Initiated",
        "confirmation_code": ""
    }

    # Step 1: Run graph until breakpoint
    print("\n1. Running workflow until approval breakpoint...")
    app.invoke(initial_request, config=config)

    # Step 2: Inspect paused state
    snapshot = app.get_state(config)
    print(f"\n2. Workflow Interrupted! Current State:")
    print(f"   Status: {snapshot.values['status']}")
    print(f"   Amount to transfer: ${snapshot.values['amount']}")
    print(f"   Recipient: {snapshot.values['recipient']}")
    print(f"   Next Pending Node: {snapshot.next}")

    # Step 3: Simulate Human Decision (Review & Approve)
    print("\n3. Human Operator Review: [APPROVE]")
    print("   Resuming execution with approval...")

    # Passing `None` as input signals the graph to resume from its current checkpoint
    final_result = app.invoke(None, config=config)

    print("\n4. Final Workflow Completion:")
    print(f"   Final Status: {final_result['status']}")
    print(f"   Confirmation Code: {final_result['confirmation_code']}")


if __name__ == "__main__":
    main()
