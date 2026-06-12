"""
chat.py — Phase 1 of CricBot (Gemini edition)
A command-line RAG chatbot. For every question it:
  1. Embeds the question and retrieves the top-k most similar chunks
     from the ChromaDB collection built by ingest.py
  2. Sends those chunks + the question to Google Gemini
  3. Prints a grounded answer with sources

Setup:  1. Get a free API key at https://aistudio.google.com (no card needed)
        2. export GEMINI_API_KEY="your-key-here"
Run:    python chat.py
"""

import os
import sys

import chromadb
from google import genai
from google.genai import types

TOP_K = 5                      # how many chunks to retrieve
MODEL = "gemini-2.5-flash"     # free tier; swap for a newer flash model anytime

SYSTEM_PROMPT = """You are CricBot, a helpful cricket expert assistant.

Answer the user's question using ONLY the provided context passages.
Rules:
- If the context does not contain the answer, say "I don't have enough
  information in my knowledge base to answer that" — never make facts up.
- Cite which source(s) you used at the end of your answer, like:
  Sources: [Indian Premier League], [Cricket World Cup]
- Keep answers concise and conversational."""


def retrieve(collection, question: str) -> list[dict]:
    """Return the top-k most relevant chunks for the question."""
    results = collection.query(query_texts=[question], n_results=TOP_K)
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "source": meta["source"]})
    return chunks


def build_user_message(question: str, chunks: list[dict]) -> str:
    """Assemble the retrieved context + question into one prompt."""
    context_blocks = []
    for i, c in enumerate(chunks, 1):
        context_blocks.append(f"[Passage {i} — source: {c['source']}]\n{c['text']}")
    context = "\n\n".join(context_blocks)
    return f"Context passages:\n\n{context}\n\nQuestion: {question}"


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("Please set the GEMINI_API_KEY environment variable first.\n"
                 "Get a free key at https://aistudio.google.com")

    db = chromadb.PersistentClient(path="cricbot_db")
    try:
        collection = db.get_collection("cricket")
    except Exception:
        sys.exit("No database found. Run `python ingest.py` first.")

    llm = genai.Client()  # reads GEMINI_API_KEY from the environment

    print("CricBot ready! Ask me anything about cricket. (type 'quit' to exit)\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        # --- RAG step 1: retrieve ---
        chunks = retrieve(collection, question)

        # --- RAG step 2: generate ---
        response = llm.models.generate_content(
            model=MODEL,
            contents=build_user_message(question, chunks),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=1024,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        answer = response.text

        print(f"\nCricBot: {answer}\n")


if __name__ == "__main__":
    main()
