"""
==============================================================================
01_prompts_and_models.py - LangChain Prompt Templates and Chat Models
==============================================================================

This script demonstrates:
1. Creating basic and conversational prompt templates (`PromptTemplate` & `ChatPromptTemplate`).
2. Composing messages with system and human roles.
3. Invoking Chat Models and inspecting response objects.
4. Graceful fallback when no API key is configured.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def get_model():
    """Initializes a chat model based on available API keys or falls back to a simulated model."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            print(" Using OpenAI ChatOpenAI (gpt-4o-mini)...")
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        except ImportError:
            pass

    if os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            print(" Using Google Gemini (gemini-1.5-flash)...")
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
        except ImportError:
            pass

    # Fallback to FakeListChatModel for zero-dependency local testing
    from langchain_community.chat_models.fake import FakeListChatModel
    print(" No OpenAI/Google API key found. Using simulated FakeListChatModel.")
    return FakeListChatModel(
        responses=[
            "LangChain is an open-source framework designed to simplify the creation of applications using large language models (LLMs).",
            "A key benefit is its modular component architecture: PromptTemplates, ChatModels, and OutputParsers.",
        ]
    )


def demo_simple_prompt():
    print("\n--- 1. Basic PromptTemplate ---")
    template = "Explain the concept of {topic} in {word_count} words for a {audience}."
    prompt = PromptTemplate.from_template(template)

    # Format the prompt with concrete variables
    formatted = prompt.format(topic="Recursion", word_count="20", audience="beginner programmer")
    print(f"Formatted Prompt:\n>> {formatted}")


def demo_chat_prompt_template(model):
    print("\n--- 2. ChatPromptTemplate with System & Human Roles ---")
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert computer science tutor who explains complex ideas using simple analogies."),
        ("human", "Explain {topic} in 2 sentences.")
    ])

    messages = chat_prompt.format_messages(topic="LangChain LCEL")
    print("Formatted Chat Messages:")
    for msg in messages:
        print(f"[{msg.__class__.__name__}] ({msg.type}): {msg.content}")

    print("\n--- 3. Invoking the Chat Model ---")
    response = model.invoke(messages)
    print(f"AI Response:\n{response.content}")
    if hasattr(response, "response_metadata") and response.response_metadata:
        print(f"Metadata: {response.response_metadata}")


if __name__ == "__main__":
    model = get_model()
    demo_simple_prompt()
    demo_chat_prompt_template(model)
