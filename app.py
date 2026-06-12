"""
app.py — Phase 4 of CricBot
A web chat interface for CricBot, built with Streamlit.
Same RAG pipeline as chat.py — retrieval from ChromaDB, generation with
Gemini, conversation memory — wrapped in a browser chat UI.

Run:  streamlit run app.py
"""

import os

import streamlit as st
import chromadb
from google import genai
from google.genai import types, errors

TOP_K = 5
MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are CricBot, a helpful cricket expert assistant.

Answer the user's question using ONLY the provided context passages.
Rules:
- If the context does not contain the answer, say "I don't have enough
  information in my knowledge base to answer that" — never make facts up.
- Cite which source(s) you used at the end of your answer, like:
  Sources: [Indian Premier League], [Cricket World Cup]
- Keep answers concise and conversational."""

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="CricBot", page_icon="🏏")
st.title("🏏 CricBot")
st.caption("A RAG chatbot grounded in Wikipedia cricket articles and "
           "1,200+ IPL matches from Cricsheet. Built by Akash Modili.")

# Allow the API key to come from Streamlit secrets (used when deployed)
if not os.environ.get("GEMINI_API_KEY"):
    try:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

if not os.environ.get("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY is not set. Export it in your terminal "
             "before running, or add it to Streamlit secrets.")
    st.stop()


# ---------------------------------------------------------------
# Cached resources — loaded once, reused across reruns
# ---------------------------------------------------------------
@st.cache_resource
def load_collection():
    db = chromadb.PersistentClient(path="cricbot_db")
    return db.get_collection("cricket")


@st.cache_resource
def load_llm():
    return genai.Client()


try:
    collection = load_collection()
except Exception:
    st.error("No database found. Run `python ingest.py` and "
             "`python ingest_matches.py` first.")
    st.stop()

llm = load_llm()


# ---------------------------------------------------------------
# RAG helpers (same logic as chat.py)
# ---------------------------------------------------------------
def retrieve(question: str) -> list[dict]:
    results = collection.query(query_texts=[question], n_results=TOP_K)
    return [{"text": doc, "source": meta["source"]}
            for doc, meta in zip(results["documents"][0],
                                 results["metadatas"][0])]


def build_user_message(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Passage {i} — source: {c['source']}]\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
    return f"Context passages:\n\n{context}\n\nQuestion: {question}"


# ---------------------------------------------------------------
# Chat state — survives reruns via session_state
# ---------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": str}]

# Replay the conversation so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------
# Handle a new question
# ---------------------------------------------------------------
if question := st.chat_input("Ask me anything about cricket..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Context-aware retrieval: include the previous two user questions
    past_questions = [m["content"] for m in st.session_state.messages
                      if m["role"] == "user"][:-1]
    retrieval_query = " ".join(past_questions[-2:] + [question])
    chunks = retrieve(retrieval_query)

    # Build Gemini-format history from prior turns (capped at 12 messages)
    history = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in st.session_state.messages[:-1][-12:]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            try:
                response = llm.models.generate_content(
                    model=MODEL,
                    contents=history + [types.Content(
                        role="user",
                        parts=[types.Part(
                            text=build_user_message(question, chunks))],
                    )],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        max_output_tokens=1024,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                answer = response.text or ("The model returned an empty "
                                           "response — try rephrasing.")
            except errors.APIError as e:
                if e.code == 429:
                    answer = ("Rate limit reached (free tier). "
                              "Wait about a minute and try again.")
                else:
                    answer = f"API error {e.code}: {e.message}. Try again shortly."
            except Exception as e:
                answer = (f"Something went wrong ({type(e).__name__}). "
                          "Check the connection and try again.")

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
