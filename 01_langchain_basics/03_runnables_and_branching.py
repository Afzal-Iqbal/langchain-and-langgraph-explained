"""
==============================================================================
03_runnables_and_branching.py - Advanced LCEL Runnables & Branching
==============================================================================

This script demonstrates:
1. `RunnablePassthrough`: Passing original inputs down the pipeline.
2. `RunnableParallel`: Executing multiple sub-chains in parallel (e.g., pros vs cons).
3. `RunnableLambda`: Wrapping plain Python functions directly into the pipeline.
4. `RunnableBranch`: Conditional execution routing based on input criteria.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
    RunnableBranch,
)


def get_model():
    """Initializes model or returns simulated fallback."""
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

    from langchain_community.chat_models.fake import FakeListChatModel
    return FakeListChatModel(
        responses=[
            "Pro: High developer velocity and large ecosystem.",
            "Con: Dynamic typing can cause runtime issues at scale.",
            "Combined Analysis: Python balances fast development with trade-offs.",
            "Technical Response: Python implements automatic reference counting and a generational GC.",
        ]
    )


# ---------------------------------------------------------------------------
# 1. Parallel Execution with RunnableParallel
# ---------------------------------------------------------------------------
def demo_parallel_chains(model):
    print("\n--- 1. RunnableParallel: Pros & Cons in Parallel ---")
    
    pros_prompt = ChatPromptTemplate.from_template("List 1 major advantage of {subject}.")
    cons_prompt = ChatPromptTemplate.from_template("List 1 major disadvantage of {subject}.")

    pros_chain = pros_prompt | model | StrOutputParser()
    cons_chain = cons_prompt | model | StrOutputParser()

    # Parallel execution block
    parallel_analysis = RunnableParallel({
        "pros": pros_chain,
        "cons": cons_chain,
        "original_subject": RunnablePassthrough(),  # Passes the raw input through
    })

    result = parallel_analysis.invoke({"subject": "Microservices Architecture"})
    print("Parallel execution outputs:")
    for key, value in result.items():
        print(f"[{key}]: {value}")


# ---------------------------------------------------------------------------
# 2. Custom Logic with RunnableLambda
# ---------------------------------------------------------------------------
def uppercase_cleaner(text: str) -> str:
    """A standard Python function converted to an LCEL Runnable."""
    return f"==> {text.strip().upper()} <=="


def demo_runnable_lambda():
    print("\n--- 2. RunnableLambda: Injecting Python Functions ---")
    lambda_step = RunnableLambda(uppercase_cleaner)
    output = lambda_step.invoke("  hello from langchain runnable  ")
    print(f"Transformed output: {output}")


# ---------------------------------------------------------------------------
# 3. Dynamic Routing with RunnableBranch
# ---------------------------------------------------------------------------
def demo_runnable_branch(model):
    print("\n--- 3. RunnableBranch: Dynamic Request Routing ---")

    general_prompt = ChatPromptTemplate.from_template("Give a simple 1-sentence answer for a general audience: {query}")
    technical_prompt = ChatPromptTemplate.from_template("Give a deep, technical 1-sentence answer for engineers: {query}")

    general_chain = general_prompt | model | StrOutputParser()
    technical_chain = technical_prompt | model | StrOutputParser()

    # Branch routing: condition function -> target chain
    branch = RunnableBranch(
        (lambda x: "code" in x["query"].lower() or "memory" in x["query"].lower(), technical_chain),
        general_chain  # Default fallback
    )

    query_tech = {"query": "How is memory managed in Python?"}
    query_general = {"query": "What is Python?"}

    print("Technical Query routed to technical chain:")
    print(" ", branch.invoke(query_tech))


if __name__ == "__main__":
    model = get_model()
    demo_parallel_chains(model)
    demo_runnable_lambda()
    demo_runnable_branch(model)
