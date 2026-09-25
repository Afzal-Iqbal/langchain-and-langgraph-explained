# Module 01: LangChain Fundamentals

This module covers the core building blocks of the LangChain framework.

---

## 🎯 What is LangChain?

**LangChain** is a developer framework designed to simplify building applications powered by Large Language Models (LLMs). It provides standardized interfaces, abstractions, and pipelines to connect language models to external data sources, prompts, tools, and parsers.

### The Core Problem LangChain Solves
When working with raw LLM APIs directly, you often deal with:
- Repetitive prompt string manipulation.
- Fragile output parsing (extracting raw JSON or objects from unstructured text).
- Manual orchestration of multi-step retrieval (RAG) and tool calling.
- Lack of streaming, async, and batch processing out of the box.

LangChain abstracts these through **LCEL (LangChain Expression Language)**.

---

## 🧩 Key Building Blocks

```
Input Dict / String
        │
        ▼
┌──────────────────────┐
│  ChatPromptTemplate  │  --> Structures system instructions & user variables
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│      ChatModel       │  --> Calls OpenAI, Gemini, Anthropic, or Local LLM
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│     OutputParser     │  --> Converts AIMessage to string, JSON, or Pydantic model
└──────────────────────┘
        │
        ▼
Parsed Result (str / dict / Pydantic object)
```

1. **Prompts & Models (`01_prompts_and_models.py`)**:
   - `PromptTemplate`: For simple text-in, text-out prompts.
   - `ChatPromptTemplate`: For conversational models with roles (`system`, `human`, `ai`).
   - `BaseChatModel`: Uniform interface across multiple model providers (OpenAI, Google Gemini, Ollama, Anthropic).

2. **LCEL & Output Parsers (`02_lcel_and_parsers.py`)**:
   - The pipe operator `|`: Chains components cleanly (`chain = prompt | model | parser`).
   - `StrOutputParser`: Extracts raw string from `AIMessage.content`.
   - `JsonOutputParser`: Extracts structured dictionary.
   - `PydanticOutputParser`: Validates and forces structured schema compliance using Pydantic.

3. **Runnables & Branching (`03_runnables_and_branching.py`)**:
   - `RunnablePassthrough`: Passes raw inputs through without alteration.
   - `RunnableParallel`: Executes multiple chains concurrently (e.g., summarize and extract key entities simultaneously).
   - `RunnableLambda`: Wraps any standard Python function into an LCEL runnable.
   - `RunnableBranch`: Conditional execution routing.

4. **RAG (Retrieval-Augmented Generation) (`04_simple_rag_pipeline.py`)**:
   - Document loading and chunking with `RecursiveCharacterTextSplitter`.
   - Vector storage and retrieval.
   - Feeding retrieved document context directly into the prompt using LCEL.

---

## 🚀 How to Run the Code

Make sure your virtual environment is active and dependencies are installed:
```bash
pip install -r requirements.txt
```

Run any script directly:
```bash
python 01_langchain_basics/01_prompts_and_models.py
python 01_langchain_basics/02_lcel_and_parsers.py
python 01_langchain_basics/03_runnables_and_branching.py
python 01_langchain_basics/04_simple_rag_pipeline.py
```
*(If no API key is detected in `.env`, the scripts automatically use simulated/mock LLM execution so you can inspect the data flow without configuring an API key first.)*
