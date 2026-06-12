# CricBot — Cricket RAG Chatbot (Phase 1)

A retrieval-augmented generation (RAG) chatbot that answers cricket questions
grounded in real Wikipedia content instead of LLM guesswork.

## How it works

**Indexing (offline):** `ingest.py` downloads ~20 cricket articles from
Wikipedia, splits them into overlapping ~900-character chunks, embeds each
chunk with a sentence-transformer model, and stores everything in a local
ChromaDB vector database (`./cricbot_db`).

**Query (runtime):** `chat.py` embeds your question, finds the 5 most
semantically similar chunks, and sends them with your question to Claude,
which answers using only that context and cites its sources.

## Setup (step by step)

1. Make sure you have Python 3.10+ installed: `python --version`

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        (Windows)
   source venv/bin/activate     (Mac/Linux)
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Get an Anthropic API key at https://console.anthropic.com and set it:
   ```
   set ANTHROPIC_API_KEY=sk-ant-...      (Windows cmd)
   $env:ANTHROPIC_API_KEY="sk-ant-..."   (Windows PowerShell)
   export ANTHROPIC_API_KEY="sk-ant-..." (Mac/Linux)
   ```

5. Build the knowledge base (takes a few minutes the first time while the
   embedding model downloads):
   ```
   python ingest.py
   ```

6. Chat:
   ```
   python chat.py
   ```

## Try asking

- What is the DLS method and when is it used?
- Explain the LBW rule in simple terms.
- Who won the 2023 Cricket World Cup?
- What is the difference between Test cricket and T20?

## What's next (Phases 2–4)

- **Phase 2:** Ingest ball-by-ball match data from Cricsheet.org by converting
  match JSON into natural-language summaries before embedding.
- **Phase 3:** Better citations, "I don't know" handling, hybrid
  keyword + semantic search.
- **Phase 4:** Streamlit chat UI and free deployment (Streamlit Cloud or
  Hugging Face Spaces).
