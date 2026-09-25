"""
==============================================================================
02_react_agent_executor.py - AgentExecutor with Tool Calling
==============================================================================

This script demonstrates:
1. Creating an agent using `create_tool_calling_agent`.
2. Running the agent loop via `AgentExecutor`.
3. Multi-step reasoning where the model selects, calls, and synthesizes tool outputs.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor


# ---------------------------------------------------------------------------
# 1. Define Tools for the Agent
# ---------------------------------------------------------------------------
@tool
def calculate(expression: str) -> str:
    """Evaluates a safe mathematical expression string, e.g. '25 * 4' or '100 / 5'."""
    try:
        # Safe evaluation restricted to numbers and standard math operators
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Unsupported characters in calculation."
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def search_product_inventory(product_name: str) -> str:
    """Looks up available warehouse inventory count and unit price for a given product."""
    catalog = {
        "laptop": {"stock": 14, "unit_price": 999.00},
        "monitor": {"stock": 32, "unit_price": 249.50},
        "keyboard": {"stock": 85, "unit_price": 79.99},
    }
    item = catalog.get(product_name.lower().strip())
    if not item:
        return f"Product '{product_name}' is not found in the catalog."
    return f"Product: {product_name}, In Stock: {item['stock']}, Unit Price: ${item['unit_price']}"


def get_agent_and_tools():
    tools = [calculate, search_product_inventory]
    
    # Check for live API key
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an accurate inventory and financial calculator assistant. Use tools when needed."),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ])
            agent = create_tool_calling_agent(llm, tools, prompt)
            executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            return executor, False
        except ImportError:
            pass

    return None, True


def run_simulated_agent_loop():
    """Educational walkthrough simulating what AgentExecutor does internally."""
    print("--- Simulated Agent Execution Loop (Step-by-Step) ---")
    query = "What is the total value of all monitors currently in stock?"
    print(f"User Query: '{query}'\n")

    print("[Step 1: LLM Reasoning]")
    print("  Thought: I need to check the inventory stock and unit price of 'monitor'.")
    print("  Action: search_product_inventory(product_name='monitor')")

    step1_output = search_product_inventory.invoke({"product_name": "monitor"})
    print(f"  Observation: {step1_output}\n")

    print("[Step 2: LLM Reasoning]")
    print("  Thought: We have 32 monitors at $249.50 each. Now I must calculate 32 * 249.50.")
    print("  Action: calculate(expression='32 * 249.50')")

    step2_output = calculate.invoke({"expression": "32 * 249.50"})
    print(f"  Observation: {step2_output}\n")

    print("[Step 3: Final Answer]")
    print(f"  Response: There are 32 monitors in stock at $249.50 each, giving a total value of ${float(step2_output):,.2f}.")


def main():
    executor, is_simulated = get_agent_and_tools()

    if is_simulated:
        print("Note: OPENAI_API_KEY not found. Running simulated step-by-step AgentExecutor loop.\n")
        run_simulated_agent_loop()
    else:
        print("Running live AgentExecutor with Tool Calling...\n")
        query = "How much would it cost to buy all the keyboards currently in stock?"
        result = executor.invoke({"input": query})
        print("\nFinal Result Output:")
        print(result["output"])


if __name__ == "__main__":
    main()
