"""
==============================================================================
02_prebuilt_create_react_agent.py - Prebuilt LangGraph ReAct Agent
==============================================================================

This script demonstrates:
1. Using `langgraph.prebuilt.create_react_agent` for production agent setups.
2. Passing tools, custom system prompts (`state_modifier` / `prompt`), and checkpointers.
3. Streaming agent thoughts, actions, and observations.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent


@tool
def calculate_tax(amount: float, tax_rate_percent: float) -> str:
    """Calculates tax amount and total price given an amount and tax rate percentage."""
    tax = amount * (tax_rate_percent / 100.0)
    total = amount + tax
    return f"Base: ${amount:.2f}, Tax ({tax_rate_percent}%): ${tax:.2f}, Total: ${total:.2f}"


@tool
def get_company_policy(topic: str) -> str:
    """Retrieves official policy text regarding 'returns', 'shipping', or 'discounts'."""
    policies = {
        "returns": "30-day no-questions-asked refund policy for items in original condition.",
        "shipping": "Free standard shipping on orders over $50. Express shipping is $12.",
        "discounts": "Students and educators receive 10% off with valid ID.",
    }
    return policies.get(topic.lower().strip(), "Standard company terms apply.")


def get_model():
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except ImportError:
            pass

    if os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        except ImportError:
            pass

    return None


def main():
    print("--- LangGraph Prebuilt create_react_agent ---")
    model = get_model()

    if not model:
        print("\n[Educational Overview]")
        print("`create_react_agent(model, tools, prompt=...)` is the recommended way to build standard agents.")
        print("It replaces LangChain's legacy AgentExecutor with a compiled, stateful StateGraph.")
        print("Example syntax:")
        print("  agent = create_react_agent(model, tools=[calculate_tax, get_company_policy])")
        print("  response = agent.invoke({'messages': [('user', 'What is your returns policy?')]})")
        print("\nTool output test directly:")
        print(" ", calculate_tax.invoke({"amount": 100, "tax_rate_percent": 8.5}))
        print(" ", get_company_policy.invoke({"topic": "returns"}))
        return

    tools = [calculate_tax, get_company_policy]
    system_prompt = "You are a helpful customer service representative. Always check policy and compute taxes accurately."

    # Build the prebuilt agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt
    )

    query = "A customer purchased items worth $150 in an area with 7% tax. What is their total, and what is the return policy?"
    print(f"User Query:\n{query}\n")

    print("Streaming Agent Execution:")
    for chunk in agent.stream({"messages": [HumanMessage(content=query)]}, stream_mode="updates"):
        for node_name, state_update in chunk.items():
            print(f"\n[Node Updated: {node_name}]")
            for msg in state_update.get("messages", []):
                print(f"  {msg.__class__.__name__}: {msg.content or '(tool call generated)'}")


if __name__ == "__main__":
    main()
