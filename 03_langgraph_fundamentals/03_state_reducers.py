"""
==============================================================================
03_state_reducers.py - State Reducers and Message History
==============================================================================

This script demonstrates:
1. The difference between default state overwriting and State Reducers.
2. Using `Annotated[list, operator.add]` to append items to a list across nodes.
3. Using `add_messages` reducer for conversational message streams.
"""

import operator
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# 1. State with Reducer: Annotated[list, operator.add]
# ---------------------------------------------------------------------------
class AuditLogState(TypedDict):
    # 'audit_trail' uses operator.add: any node returning a list will have its
    # elements appended to the existing list, rather than replacing it!
    audit_trail: Annotated[List[str], operator.add]
    
    # 'latest_status' has NO reducer: returning it overwrites the previous value.
    latest_status: str


def step_one_node(state: AuditLogState) -> dict:
    return {
        "audit_trail": ["Step 1: User authentication verified."],
        "latest_status": "Authenticated"
    }


def step_two_node(state: AuditLogState) -> dict:
    return {
        "audit_trail": ["Step 2: Database transaction committed."],
        "latest_status": "Committed"
    }


def step_three_node(state: AuditLogState) -> dict:
    return {
        "audit_trail": ["Step 3: Receipt email dispatched."],
        "latest_status": "Completed"
    }


def demo_operator_add_reducer():
    print("--- 1. Testing operator.add List Reducer ---")
    builder = StateGraph(AuditLogState)
    builder.add_node("step1", step_one_node)
    builder.add_node("step2", step_two_node)
    builder.add_node("step3", step_three_node)

    builder.add_edge(START, "step1")
    builder.add_edge("step1", "step2")
    builder.add_edge("step2", "step3")
    builder.add_edge("step3", END)

    app = builder.compile()
    result = app.invoke({"audit_trail": ["Step 0: Request initiated"], "latest_status": "Pending"})

    print(f"Final Status (Overwritten): {result['latest_status']}")
    print("Accumulated Audit Trail (Appended via Reducer):")
    for log in result["audit_trail"]:
        print(f"  • {log}")


# ---------------------------------------------------------------------------
# 2. Chat Message Reducer: add_messages
# ---------------------------------------------------------------------------
class ConversationState(TypedDict):
    # 'add_messages' handles deduplication, message ID tracking, and appending
    messages: Annotated[List[BaseMessage], add_messages]


def mock_ai_reply_node(state: ConversationState) -> dict:
    # Appends an AIMessage to the list
    return {
        "messages": [AIMessage(content="Hello! I am your AI assistant. How can I help you today?")]
    }


def demo_add_messages_reducer():
    print("\n--- 2. Testing add_messages Reducer ---")
    builder = StateGraph(ConversationState)
    builder.add_node("agent", mock_ai_reply_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)

    app = builder.compile()

    initial_messages = [HumanMessage(content="Hi there!")]
    result = app.invoke({"messages": initial_messages})

    print(f"Total messages in history: {len(result['messages'])}")
    for msg in result["messages"]:
        print(f"  [{msg.__class__.__name__}]: {msg.content}")


if __name__ == "__main__":
    demo_operator_add_reducer()
    demo_add_messages_reducer()
