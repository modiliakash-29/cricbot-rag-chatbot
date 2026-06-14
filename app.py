"""
app.py — CricBot 2.0
Upgrades over v1:
  • Tiered answering: verified knowledge base first; clearly-labeled
    general cricket knowledge as fallback (no more flat refusals)
  • Optional live web search (Gemini Google Search grounding) for
    recent matches and news
  • New interface: dark cricket theme, sidebar with sample questions,
    clear-chat, message avatars, and a "retrieved sources" expander

Run:  python -m streamlit run app.py
"""

import os

import streamlit as st
import chromadb
from google import genai
from google.genai import types, errors

TOP_K = 5
MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are CricBot, an expert cricket assistant.

You answer using TWO tiers of knowledge:

TIER 1 — VERIFIED: the context passages provided with each question
(from a curated database of Wikipedia articles and 1,200+ IPL matches).
Prefer this. When your answer comes from the passages, end with:
Sources: [source name], [source name]

TIER 2 — GENERAL KNOWLEDGE: if the passages do not contain the answer
but you reliably know it from general cricket knowledge, answer anyway,
and you MUST end with this exact label instead of sources:
⚠️ From general knowledge — not verified against my database.

Rules:
- Never mix the two labels in one answer.
- If you are not confident even from general knowledge, say you are
  not sure rather than guessing. Never invent statistics.
- Only answer cricket-related questions; politely decline others.
- Keep answers concise and conversational."""

st.set_page_config(page_title="CricBot", page_icon="🏏", layout="wide")

# ---------------------------------------------------------------
# Custom styling on top of the theme
# ---------------------------------------------------------------
st.markdown("""
<style>
/* Hide Streamlit's default chrome (toolbar, main menu, footer, header) */
[data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}
[data-testid="stDecoration"] {display: none;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.cric-header {
    background: linear-gradient(90deg, #14331f 0%, #1d4a2c 55%, #b8860b 130%);
    padding: 1.4rem 1.8rem;
    border-radius: 14px;
    margin-bottom: 1rem;
    border: 1px solid #2e5c3c;
}
.cric-header h1 {
    color: #f5b942;
    margin: 0;
    font-size: 2.1rem;
}
.cric-header p {
    color: #cfe3d4;
    margin: 0.35rem 0 0 0;
    font-size: 0.95rem;
}
div[data-testid="stChatMessage"] {
    border-radius: 12px;
    margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Secrets / API key
# ---------------------------------------------------------------
if not os.environ.get("GEMINI_API_KEY"):
    try:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

if not os.environ.get("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY is not set. Export it in your terminal "
             "or add it to Streamlit secrets.")
    st.stop()


# ---------------------------------------------------------------
# Cached resources
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
# RAG helpers
# ---------------------------------------------------------------
def retrieve(query: str) -> list[dict]:
    results = collection.query(query_texts=[query], n_results=TOP_K)
    return [{"text": doc, "source": meta["source"]}
            for doc, meta in zip(results["documents"][0],
                                 results["metadatas"][0])]


def build_user_message(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Passage {i} — source: {c['source']}]\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
    return f"Context passages:\n\n{context}\n\nQuestion: {question}"


def rewrite_query(question: str, history_msgs: list[dict]) -> str:
    """
    Turn a follow-up ("what about his international career?") into a
    standalone search query ("Suresh Raina international cricket career")
    using recent conversation. This fixes topic drift: without it,
    retrieval can latch onto whoever the embedding model finds most
    salient rather than who the conversation is actually about.
    Falls back to the raw question if anything goes wrong.
    """
    if not history_msgs:
        return question
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history_msgs[-4:])
    prompt = (
        "Given the conversation, rewrite the user's latest question as a "
        "standalone search query that includes the specific player, team, "
        "season, or topic being discussed. Output ONLY the query, nothing "
        f"else.\n\nConversation:\n{convo}\n\nLatest question: {question}\n\n"
        "Standalone search query:"
    )
    try:
        resp = llm.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=60,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        rewritten = (resp.text or "").strip()
        return rewritten if rewritten else question
    except Exception:
        return question


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
SAMPLES = [
    "Who won the IPL in 2020?",
    "Who won the Orange Cap in 2016?",
    "Explain the LBW rule in simple terms",
    "What is the DLS method?",
    "Who has won the most IPL titles?",
]

with st.sidebar:
    st.markdown("## 🏏 CricBot")
    st.caption("RAG chatbot over Wikipedia cricket articles + ball-by-ball "
               "data for 1,200+ IPL matches. Built by **Akash Modili**.")

    st.divider()
    st.markdown("**Try asking:**")
    for q in SAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q
            st.rerun()

    st.divider()
    use_web = st.toggle(
        "🌐 Live web search",
        value=False,
        help="Lets CricBot search the web for recent matches and news "
             "(uses Gemini's Google Search grounding).",
    )

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"Knowledge base: {collection.count():,} documents")
    st.caption("[Source code](https://github.com/modiliakash-29/"
               "cricbot-rag-chatbot)")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div class="cric-header">
  <h1>🏏 CricBot</h1>
  <p>Ask me anything about cricket — rules, history, IPL matches, stats.
  Database-verified answers are cited; everything else is clearly labeled.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Chat state + replay
# ---------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

AVATARS = {"user": "🧑", "assistant": "🏏"}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(msg["content"])
        if msg.get("chunks"):
            with st.expander("🔎 Retrieved passages"):
                for c in msg["chunks"]:
                    st.markdown(f"**{c['source']}** — {c['text'][:300]}…")

# ---------------------------------------------------------------
# Input: chat box or a clicked sample question
# ---------------------------------------------------------------
question = st.chat_input("Ask me anything about cricket...")
if st.session_state.get("pending"):
    question = st.session_state.pop("pending")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    # Resolve follow-ups into a standalone query before retrieval.
    # "what about his international career?" → "Suresh Raina international
    # career" — so the search finds the right person, not whoever the
    # embedding model finds most salient.
    prior = st.session_state.messages[:-1]
    search_query = rewrite_query(question, prior)
    chunks = retrieve(search_query)

    # Conversation history for the model (capped)
    history = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in st.session_state.messages[:-1][-12:]
    ]

    config_kwargs = dict(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=1024,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    # Optional web grounding for recent info
    if use_web:
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Checking the knowledge base..."):
            try:
                response = llm.models.generate_content(
                    model=MODEL,
                    contents=history + [types.Content(
                        role="user",
                        parts=[types.Part(
                            text=build_user_message(question, chunks))],
                    )],
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                answer = response.text or ("The model returned an empty "
                                           "response — try rephrasing.")
            except errors.APIError as e:
                if e.code == 429:
                    answer = ("Rate limit reached (free tier). "
                              "Wait about a minute and try again.")
                else:
                    answer = (f"API error {e.code}: {e.message}. "
                              "Try again shortly.")
            except Exception as e:
                answer = (f"Something went wrong ({type(e).__name__}). "
                          "Check the connection and try again.")

        st.markdown(answer)
        with st.expander("🔎 Retrieved passages"):
            for c in chunks:
                st.markdown(f"**{c['source']}** — {c['text'][:300]}…")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "chunks": chunks})
