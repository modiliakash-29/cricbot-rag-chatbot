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
import re

import streamlit as st
import chromadb
from rank_bm25 import BM25Okapi
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

st.set_page_config(page_title="CricBot", page_icon="🏏", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------
# Custom styling on top of the theme
# ---------------------------------------------------------------
st.markdown("""
<style>
/* Faded cricket-stadium backdrop (free Unsplash image), darkened so
   the dark theme and text stay readable. Fixed so it doesn't scroll. */
[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(rgba(6,12,8,0.93), rgba(6,12,8,0.97)),
        url("https://images.unsplash.com/photo-1531415074968-036ba1b575da?auto=format&fit=crop&w=1600&q=70");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] {
    background: rgba(10,20,13,0.92);
    backdrop-filter: blur(2px);
}
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
/* Landing state — informational cards (NOT clickable) */
.cric-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 0.9rem;
    margin: 0.5rem 0 1.4rem 0;
}
.cric-card {
    background: transparent;
    border: none;
    border-left: 2px solid #3a6b48;
    border-radius: 0;
    padding: 0.2rem 0 0.2rem 0.9rem;
    cursor: default;
}
.cric-card .ico { font-size: 1.25rem; opacity: 0.9; }
.cric-card h4 {
    color: #cBe0d0;
    margin: 0.3rem 0 0.2rem 0;
    font-size: 0.92rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.cric-card p {
    color: #8ea596;
    margin: 0;
    font-size: 0.82rem;
    line-height: 1.35;
}
.cric-welcome {
    color: #e8efe9;
    font-size: 1.15rem;
    font-weight: 600;
    margin: 0.4rem 0 0.2rem 0;
}
.cric-sub { color: #93a89a; font-size: 0.9rem; margin-bottom: 1rem; }
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


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens for BM25 keyword matching."""
    return re.findall(r"[a-z0-9]+", text.lower())


@st.cache_resource
def load_bm25(_collection):
    """
    Build a BM25 keyword index over the entire corpus, once.
    BM25 ranks documents by exact-term overlap with the query — the
    complement to semantic search, which can miss exact names and
    abbreviations. Cached so it's built a single time per session.
    The leading underscore on _collection tells Streamlit not to try
    to hash the Chroma object (it isn't hashable).
    """
    data = _collection.get()  # all documents + metadata
    docs = data["documents"]
    metas = data["metadatas"]
    tokenized = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, docs, metas


try:
    collection = load_collection()
except Exception:
    st.error("No database found. Run `python ingest.py` and "
             "`python ingest_matches.py` first.")
    st.stop()

llm = load_llm()
bm25, bm25_docs, bm25_metas = load_bm25(collection)


# ---------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------
def retrieve(query: str) -> list[dict]:
    """
    Hybrid retrieval: combine semantic (vector) and keyword (BM25)
    search, then fuse with Reciprocal Rank Fusion (RRF).

    Why hybrid? Semantic search captures meaning/paraphrase but can
    miss exact tokens (abbreviations, specific names). BM25 nails exact
    terms but misses paraphrase. RRF rewards documents that rank well
    in EITHER list, so we get both strengths.

    RRF score for a doc = sum over each ranked list of 1/(k + rank).
    k=60 is the standard constant; it damps the influence of any single
    list so no one method dominates.
    """
    POOL = 20   # candidates to pull from each method
    K = 60      # RRF damping constant

    # --- Semantic ranking (ChromaDB) ---
    sem = collection.query(query_texts=[query], n_results=POOL)
    sem_docs = sem["documents"][0]
    sem_metas = sem["metadatas"][0]

    # --- Keyword ranking (BM25 over full corpus) ---
    scores = bm25.get_scores(_tokenize(query))
    top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:POOL]

    # --- Fuse with RRF, keyed by document text ---
    rrf: dict[str, float] = {}
    lookup: dict[str, dict] = {}

    for rank, (doc, meta) in enumerate(zip(sem_docs, sem_metas)):
        rrf[doc] = rrf.get(doc, 0.0) + 1.0 / (K + rank)
        lookup[doc] = {"text": doc, "source": meta["source"]}

    for rank, i in enumerate(top_idx):
        doc = bm25_docs[i]
        rrf[doc] = rrf.get(doc, 0.0) + 1.0 / (K + rank)
        lookup[doc] = {"text": doc, "source": bm25_metas[i]["source"]}

    # Top TOP_K after fusion
    best = sorted(rrf.items(), key=lambda x: -x[1])[:TOP_K]
    return [lookup[doc] for doc, _ in best]


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
  <p>Your cricket brain, grounded in real data.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Chat state + replay
# ---------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

AVATARS = {"user": "🧑", "assistant": "🏏"}

# Landing state — shown only before the first message. Fills the empty
# space with what CricBot can do and one-tap starter questions.
if not st.session_state.messages:
    st.markdown(
        '<div class="cric-welcome">Welcome 👋 What would you like to know?</div>'
        '<div class="cric-sub">CricBot answers from a verified database of '
        f'{collection.count():,} documents — Wikipedia cricket knowledge plus '
        'ball-by-ball records of 1,200+ IPL matches.</div>'
        '<div style="color:#7e948a;font-size:0.8rem;text-transform:uppercase;'
        'letter-spacing:0.08em;margin-bottom:0.5rem;">What I can answer</div>'
        '<div class="cric-cards">'
        '<div class="cric-card"><div class="ico">📜</div><h4>Rules &amp; History</h4>'
        '<p>LBW, DLS, formats, the origins of the game.</p></div>'
        '<div class="cric-card"><div class="ico">🏆</div><h4>IPL Results</h4>'
        '<p>Winners, finals, and season summaries by year.</p></div>'
        '<div class="cric-card"><div class="ico">📊</div><h4>Stats &amp; Records</h4>'
        '<p>Orange Cap, Purple Cap, player and match numbers.</p></div>'
        '<div class="cric-card"><div class="ico">🧠</div><h4>Follow-ups</h4>'
        '<p>Ask naturally — it remembers the conversation.</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Start with one of these:**")
    cols = st.columns(2)
    for i, q in enumerate(SAMPLES):
        if cols[i % 2].button(q, use_container_width=True, key=f"land_{i}"):
            st.session_state.pending = q
            st.rerun()

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
question = st.chat_input("What do you want to know about cricket?")
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
