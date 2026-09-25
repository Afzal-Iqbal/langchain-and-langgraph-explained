"""
==============================================================================
04_simple_rag_pipeline.py - Retrieval-Augmented Generation (RAG) with LCEL
==============================================================================

This script demonstrates:
1. Creating and chunking text documents with `RecursiveCharacterTextSplitter`.
2. Initializing an in-memory document retriever.
3. Building an LCEL RAG chain:
   {context: retriever | format_docs, question: RunnablePassthrough()} | prompt | model | StrOutputParser()
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
            "Based on the provided documentation, LangGraph enables cyclical graphs and state persistence, whereas standard LangChain chains are directed acyclic graphs (DAGs)."
        ]
    )


# Sample knowledge base documents
SAMPLE_KNOWLEDGE_BASE = [
    Document(
        page_content="LangChain is ideal for linear workflows, prompt sequencing, and standard RAG pipelines. It represents sequences as DAGs.",
        metadata={"source": "langchain_docs.md", "topic": "chains"}
    ),
    Document(
        page_content="LangGraph extends LangChain by introducing cyclic graph execution, first-class state management, and built-in human-in-the-loop checkpoints.",
        metadata={"source": "langgraph_docs.md", "topic": "agents"}
    ),
    Document(
        page_content="Checkpoints in LangGraph allow thread-level persistence, enabling features like time travel, rewind, and interactive debugging.",
        metadata={"source": "checkpoints_docs.md", "topic": "persistence"}
    )
]


class SimpleInMemoryRetriever:
    """A lightweight, zero-dependency retriever for educational demonstration."""
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, query: str):
        query_words = set(query.lower().split())
        scored = []
        for doc in self.documents:
            doc_words = set(doc.page_content.lower().split())
            score = len(query_words.intersection(doc_words))
            scored.append((score, doc))
        # Sort by match score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored if score > 0][:2]


def format_docs(docs):
    """Formats retrieved documents into a clean string block for the prompt."""
    return "\n\n".join(f"[Source: {d.metadata.get('source')}]: {d.page_content}" for d in docs)


def main():
    print("--- Simple In-Memory RAG Pipeline with LCEL ---")

    # 1. Text Splitting Demonstration
    raw_text = (
        "Artificial Intelligence agents require memory and state to solve complex tasks. "
        "LangGraph provides state persistence and cyclical loops for agentic systems. "
        "Standard LangChain chains execute in a single forward pass."
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=70, chunk_overlap=15)
    chunks = splitter.split_text(raw_text)
    print(f"Original text split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk}")

    # 2. Setup Retriever
    retriever = SimpleInMemoryRetriever(SAMPLE_KNOWLEDGE_BASE)

    # 3. Define RAG Prompt
    rag_prompt = ChatPromptTemplate.from_template("""
You are an expert technical assistant. Answer the user's question using ONLY the provided context.

Context:
{context}

Question:
{question}

Answer concisely:
""")

    model = get_model()

    # 4. Construct LCEL RAG Chain
    # The dictionary automatically pipes the question through the retriever and formats docs
    rag_chain = (
        {
            "context": retriever.invoke | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        if False else  # Using clean direct dictionary composition:
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
        }
        | rag_prompt
        | model
        | StrOutputParser()
    )

    query = "How does LangGraph differ from standard LangChain regarding loops and state?"
    print(f"\nUser Query: '{query}'")

    retrieved = retriever.invoke(query)
    print("\nRetrieved Context Documents:")
    print(format_docs(retrieved))

    print("\nRAG Generated Answer:")
    answer = rag_chain.invoke({"question": query})
    print(answer)


if __name__ == "__main__":
    main()
