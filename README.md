# 🏏 CricBot — Cricket RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers cricket questions
grounded in real data — Wikipedia articles plus ball-by-ball records of
**1,200+ IPL matches** — instead of LLM guesswork.

**🔗 Live demo:** https://cricbot-akash.streamlit.app/
## What it can do

- Answer rules, history, and player questions from Wikipedia knowledge
- Answer match questions ("Who won the 2019 IPL final? Where was it played?")
  from Cricsheet ball-by-ball data
- Answer season aggregates computed from raw deliveries — Orange Cap,
  Purple Cap, champions by year
- Hold a conversation: follow-ups like "and who was player of the match?"
  resolve against chat history
- Refuse to guess: if the knowledge base lacks the answer, it says so,
  and every answer cites its sources

## Architecture

```
INDEXING (offline)
Wikipedia articles ──┐
                     ├─► chunk ─► embed (all-MiniLM-L6-v2) ─► ChromaDB
Cricsheet JSON ──────┤
(1,243 matches)      └─► transform to natural-language match
                         summaries + computed season aggregates

QUERY (runtime)
question ─► context-aware retrieval (top-5 chunks) ─► Gemini 2.5 Flash
            (recent questions included)               (grounded prompt)
                                                   └─► cited answer
```

Key engineering decisions:

- **Structured → narrative transformation.** Ball-by-ball JSON doesn't embed
  meaningfully, so each match is converted to a natural-language summary,
  and per-season stats (most runs/wickets, champions) are computed and
  written as their own documents — data shaped at every level users ask at.
- **Retrieval vocabulary engineering.** Documents deliberately include the
  phrasings users search with ("won the IPL title", "Orange Cap",
  "leading run-scorer") so embedding search connects question to answer.
- **Context-aware retrieval.** Follow-up questions carry no searchable
  topic words, so retrieval queries include the previous two questions.
- **Graceful degradation.** API rate limits and outages produce friendly
  messages, not crashes (caught a real 503 in testing).

## Stack

Python · ChromaDB (vector store) · sentence-transformers embeddings ·
Google Gemini API · Streamlit · Cricsheet.org open data · Wikipedia API

## Run it locally

```bash
git clone https://github.com/modiliakash-29/cricbot-rag-chatbot.git
cd cricbot-rag-chatbot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export GEMINI_API_KEY="your-key"   # free at aistudio.google.com

python ingest.py           # build Wikipedia knowledge base
python ingest_matches.py   # add IPL match data + season aggregates
python chat.py             # terminal chat, or:
python -m streamlit run app.py   # web UI
```

## Project structure

| File | Purpose |
|---|---|
| `ingest.py` | Fetch + chunk + embed Wikipedia cricket articles |
| `ingest_matches.py` | Download Cricsheet data, transform JSON → summaries, compute season aggregates |
| `chat.py` | Terminal RAG chat with memory and error handling |
| `app.py` | Streamlit web UI (deployed version) |

## Data

Match data from [Cricsheet](https://cricsheet.org) (open, CC-licensed
ball-by-ball records). Stats are computed from this data and may differ
marginally from official records. Encyclopedia content from Wikipedia.

---

Built by **Akash Modili** — [GitHub](https://github.com/modiliakash-29) ·
[LinkedIn]([(https://www.linkedin.com/in/akash-modili/)
