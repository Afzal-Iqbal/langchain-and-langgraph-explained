"""
==============================================================================
01_langgraph_react_agent.py - Custom Cyclical ReAct Agent from Scratch
==============================================================================

This script demonstrates:
1. Creating a stateful ReAct agent using `StateGraph`.
2. Using `ToolNode` to execute tool calls.
3. Using `tools_condition` to route dynamically between `tools` and `END`.
4. Closing the cycle: `tools` -> `agent` until reasoning concludes.
"""

import os
from typing import Annotated, List
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ---------------------------------------------------------------------------
# 1. Define Tools
# ---------------------------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Get current weather forecast for a specified city."""
    weather_data = {
        "london": "Partly cloudy, 18°C with light breeze.",
        "tokyo": "Sunny, 26°C with moderate humidity.",
        "san francisco": "Foggy, 15°C with cool winds.",
    }
    return weather_data.get(city.lower().strip(), f"Weather for '{city}' is currently sunny and 22°C.")


@tool
def get_currency_exchange_rate(from_curr: str, to_curr: str) -> str:
    """Retrieves current exchange rate between two currency codes (e.g. USD, EUR, GBP, JPY)."""
    rates = {
        ("USD", "EUR"): 0.92,
        ("USD", "GBP"): 0.78,
        ("USD", "JPY"): 155.40,
        ("EUR", "USD"): 1.09,
    }
    key = (from_curr.upper(), to_curr.upper())
    rate = rates.get(key, 1.25)
    return f"1 {from_curr.upper()} = {rate} {to_curr.upper()}"


# ---------------------------------------------------------------------------
# 2. Graph State
# ---------------------------------------------------------------------------
class AgentState(dict):
    """State containing conversation messages with add_messages reducer."""
    messages: Annotated[List[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# 3. Model & Node Definition
# ---------------------------------------------------------------------------
tools = [get_weather, get_currency_exchange_rate]


def get_llm():
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


def build_custom_react_agent():
    llm = get_llm()

    # Agent Node: invokes LLM with tools bound
    def agent_node(state):
        print("  [Agent Node] Calling model...")
        if llm:
            model_with_tools = llm.bind_tools(tools)
            response = model_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        else:
            # Simulated tool call response when no API key is provided
            print("  (Simulating tool call to get_weather)")
            ai_msg = AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_weather",
                    "args": {"city": "Tokyo"},
                    "id": "call_tokyo_weather_001"
                }]
            )
            return {"messages": [ai_msg]}

    # Tool execution node
    tool_node = ToolNode(tools)

    # Build the StateGraph
    builder = StateGraph(dict)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    # Start -> Agent
    builder.add_edge(START, "agent")

    # Conditional edge: Does the agent want to call a tool or finish?
    builder.add_conditional_edges("agent", tools_condition)

    # Cyclical edge: After running tools, ALWAYS loop back to agent!
    builder.add_edge("tools", "agent")

    return builder.compile()


def main():
    print("--- Running Custom LangGraph ReAct Agent ---")
    app = build_custom_react_agent()

    input_message = HumanMessage(content="What's the weather like in Tokyo right now?")
    print(f"User Request: {input_message.content}\n")

    # If simulated (no API key), we demonstrate the step-by-step stream
    if not get_llm():
        print("[Simulated Environment Demo]")
        print("1. START -> 'agent' emits tool call to 'get_weather'")
        print("2. 'tools_condition' detects tool_call -> routes to 'tools' node")
        print("3. 'tools' executes get_weather(city='Tokyo') -> returns ToolMessage")
        print("4. Cyclical edge routes back to 'agent' -> agent formats final answer -> END")
        print(f"Tool Output: {get_weather.invoke({'city': 'Tokyo'})}")
        return

    # Live execution with real LLM
    for step in app.stream({"messages": [input_message]}, stream_mode="values"):
        last_msg = step["messages"][-1]
        print(f"\n[{last_msg.__class__.__name__}]: {last_msg.content}")
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            print(f"Tool Calls: {last_msg.tool_calls}")


if __name__ == "__main__":
    main()
