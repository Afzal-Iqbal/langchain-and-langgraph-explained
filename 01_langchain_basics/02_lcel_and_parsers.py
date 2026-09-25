"""
==============================================================================
02_lcel_and_parsers.py - LangChain Expression Language (LCEL) & Output Parsers
==============================================================================

This script demonstrates:
1. Composing an LCEL chain using the pipe operator (|).
2. Converting raw model outputs with `StrOutputParser`.
3. Extracting JSON dictionaries with `JsonOutputParser`.
4. Enforcing type safety and schema validation using `PydanticOutputParser`.
"""

import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser


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
    # Mock responses for string, JSON, and Pydantic tests
    return FakeListChatModel(
        responses=[
            "Paris is known as the City of Light.",
            '{"name": "Python", "creator": "Guido van Rossum", "year": 1991}',
            '{"title": "Clean Code", "author": "Robert C. Martin", "rating": 4.7, "summary": "A handbook of agile software craftsmanship."}'
        ]
    )


# ---------------------------------------------------------------------------
# 1. String Output Parser Demo
# ---------------------------------------------------------------------------
def demo_str_parser(model):
    print("\n--- 1. LCEL with StrOutputParser ---")
    prompt = ChatPromptTemplate.from_template("What is a fun nickname for the city of {city}?")
    
    # LCEL Composition: prompt | model | parser
    chain = prompt | model | StrOutputParser()
    
    result = chain.invoke({"city": "Paris"})
    print(f"Output (type: {type(result).__name__}): {result}")


# ---------------------------------------------------------------------------
# 2. JSON Output Parser Demo
# ---------------------------------------------------------------------------
def demo_json_parser(model):
    print("\n--- 2. LCEL with JsonOutputParser ---")
    json_parser = JsonOutputParser()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a tech historian. Respond ONLY with valid JSON. {format_instructions}"),
        ("human", "Tell me about the programming language {language}.")
    ]).partial(format_instructions=json_parser.get_format_instructions())

    chain = prompt | model | json_parser
    result = chain.invoke({"language": "Python"})
    print(f"Parsed JSON Dictionary (type: {type(result).__name__}):")
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# 3. Pydantic Output Parser Demo
# ---------------------------------------------------------------------------
class BookReview(BaseModel):
    title: str = Field(description="Title of the book")
    author: str = Field(description="Author's name")
    rating: float = Field(description="Rating from 1.0 to 5.0")
    summary: str = Field(description="A brief 1-sentence synopsis")


def demo_pydantic_parser(model):
    print("\n--- 3. LCEL with PydanticOutputParser ---")
    pydantic_parser = PydanticOutputParser(pydantic_object=BookReview)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a book reviewer. Respond according to the schema:\n{format_instructions}"),
        ("human", "Provide details for the book '{book_title}'.")
    ]).partial(format_instructions=pydantic_parser.get_format_instructions())

    chain = prompt | model | pydantic_parser
    review: BookReview = chain.invoke({"book_title": "Clean Code"})
    print(f"Parsed Pydantic Object (type: {type(review).__name__}):")
    print(f"  Title:   {review.title}")
    print(f"  Author:  {review.author}")
    print(f"  Rating:  {review.rating}/5.0")
    print(f"  Summary: {review.summary}")


if __name__ == "__main__":
    model = get_model()
    demo_str_parser(model)
    demo_json_parser(model)
    demo_pydantic_parser(model)
