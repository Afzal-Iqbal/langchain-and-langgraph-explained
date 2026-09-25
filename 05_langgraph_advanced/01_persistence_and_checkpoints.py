"""
==============================================================================
01_persistence_and_checkpoints.py - State Persistence and MemorySaver
==============================================================================

This script demonstrates:
1. Attaching a `MemorySaver` checkpointer to a StateGraph.
2. Managing multi-turn sessions using `thread_id`.
3. Verifying that state persists across multiple invocations for the same thread.
4. Thread isolation (thread A doesn't see thread B's memory).
5. Inspecting state snapshots via `get_state()` and `get_state_history()`.
"""

from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def chatbot_node(state: ChatState) -> dict:
    """A simulated conversational node that remembers user name and context."""
    user_msgs = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
    latest_msg = user_msgs[-1] if user_msgs else ""
    
    # Check accumulated history
    full_text = " ".join(user_msgs)
    if "my name is" in latest_msg.lower():
        name = latest_msg.lower().split("my name is")[-1].strip().capitalize()
        reply = f"Nice to meet you, {name}! I will remember your name."
    elif "what is my name" in latest_msg.lower():
        if "afzal" in full_text.lower():
            reply = "Your name is Afzal! You told me earlier in this thread."
        else:
            reply = "You haven't told me your name yet in this thread."
    else:
        reply = f"Understood. Message received. (Total conversation history length: {len(state['messages']) + 1})"

    return {"messages": [AIMessage(content=reply)]}


def build_persisted_graph():
    # 1. Initialize in-memory checkpointer
    checkpointer = MemorySaver()

    # 2. Build graph
    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    # 3. Compile WITH checkpointer!
    return builder.compile(checkpointer=checkpointer)


def main():
    print("--- 1. Initializing Graph with MemorySaver Checkpointer ---")
    app = build_persisted_graph()

    # Session 1: Thread Alpha
    config_thread_1 = {"configurable": {"thread_id": "session_alpha"}}
    print("\n[Thread 'session_alpha' - Turn 1]")
    res1 = app.invoke({"messages": [HumanMessage(content="Hello! My name is Afzal.")]}, config=config_thread_1)
    print(f"User:  'Hello! My name is Afzal.'")
    print(f"Bot:   '{res1['messages'][-1].content}'")

    print("\n[Thread 'session_alpha' - Turn 2 (Testing Memory)]")
    res2 = app.invoke({"messages": [HumanMessage(content="Do you know what is my name?")]}, config=config_thread_1)
    print(f"User:  'Do you know what is my name?'")
    print(f"Bot:   '{res2['messages'][-1].content}'")

    # Session 2: Thread Beta (Demonstrating Thread Isolation)
    config_thread_2 = {"configurable": {"thread_id": "session_beta"}}
    print("\n[Thread 'session_beta' - Turn 1 (Separate User / Clean Slate)]")
    res3 = app.invoke({"messages": [HumanMessage(content="What is my name?")]}, config=config_thread_2)
    print(f"User:  'What is my name?'")
    print(f"Bot:   '{res3['messages'][-1].content}'")

    # Inspecting Checkpoint State
    print("\n--- 2. Inspecting Saved State Snapshot ---")
    current_snapshot = app.get_state(config_thread_1)
    print(f"Thread 'session_alpha' current values:")
    print(f"  Total messages saved: {len(current_snapshot.values['messages'])}")
    print(f"  Next executable node: {current_snapshot.next}")


if __name__ == "__main__":
    main()
