"""
==============================================================================
02_conditional_routing.py - Conditional Edges and Dynamic Branching
==============================================================================

This script demonstrates:
1. Creating routing decision functions based on state values.
2. Using `add_conditional_edges` to direct execution to different nodes.
3. Building an automated triage router (Billing vs Tech Support vs General).
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------------------------
class TicketState(TypedDict):
    ticket_id: str
    user_message: str
    category: str
    agent_response: str
    urgency_level: str


# ---------------------------------------------------------------------------
# 2. Node Functions
# ---------------------------------------------------------------------------
def classify_ticket_node(state: TicketState) -> dict:
    """Classifies the ticket category based on keywords."""
    msg = state["user_message"].lower()
    print(f"  [classify_ticket_node] Analyzing ticket #{state['ticket_id']}: '{msg}'")

    if any(k in msg for k in ["refund", "invoice", "charge", "payment", "billing"]):
        return {"category": "billing"}
    elif any(k in msg for k in ["crash", "bug", "error", "broken", "500"]):
        return {"category": "technical"}
    else:
        return {"category": "general"}


def billing_support_node(state: TicketState) -> dict:
    print("  -> Routing to: [Billing Specialist Node]")
    return {
        "agent_response": "Your billing query has been forwarded to our finance department.",
        "urgency_level": "Medium"
    }


def technical_support_node(state: TicketState) -> dict:
    print("  -> Routing to: [Technical Support Node]")
    return {
        "agent_response": "An incident ticket has been created and assigned to on-call engineering.",
        "urgency_level": "High"
    }


def general_support_node(state: TicketState) -> dict:
    print("  -> Routing to: [General Helpdesk Node]")
    return {
        "agent_response": "Thank you for reaching out! A customer representative will reply shortly.",
        "urgency_level": "Standard"
    }


# ---------------------------------------------------------------------------
# 3. Router Function for Conditional Edge
# ---------------------------------------------------------------------------
def route_ticket(state: TicketState) -> Literal["billing", "technical", "general"]:
    """Returns the destination node name based on the category determined in state."""
    category = state.get("category")
    print(f"  [route_ticket decision]: category is '{category}'")
    if category == "billing":
        return "billing"
    elif category == "technical":
        return "technical"
    else:
        return "general"


# ---------------------------------------------------------------------------
# 4. Construct Graph with Conditional Edges
# ---------------------------------------------------------------------------
def build_ticket_routing_graph():
    builder = StateGraph(TicketState)

    # Register all nodes
    builder.add_node("classify", classify_ticket_node)
    builder.add_node("billing_node", billing_support_node)
    builder.add_node("technical_node", technical_support_node)
    builder.add_node("general_node", general_support_node)

    # Fixed edge from START to classify
    builder.add_edge(START, "classify")

    # Conditional edge branching from 'classify'
    builder.add_conditional_edges(
        "classify",
        route_ticket,
        {
            "billing": "billing_node",
            "technical": "technical_node",
            "general": "general_node",
        }
    )

    # All specialist nodes converge to END
    builder.add_edge("billing_node", END)
    builder.add_edge("technical_node", END)
    builder.add_edge("general_node", END)

    return builder.compile()


def main():
    app = build_ticket_routing_graph()

    test_queries = [
        {"ticket_id": "101", "user_message": "My card was charged twice for the renewal."},
        {"ticket_id": "102", "user_message": "The web app returns a 500 internal server error crash."},
        {"ticket_id": "103", "user_message": "What are your business hours during holidays?"},
    ]

    for item in test_queries:
        print(f"\n==================== Ticket #{item['ticket_id']} ====================")
        result = app.invoke(item)
        print(f"Resolution Category: {result['category']}")
        print(f"Assigned Urgency:    {result['urgency_level']}")
        print(f"Response Sent:       {result['agent_response']}")


if __name__ == "__main__":
    main()
