"""
ingest.py — Phase 1 of CricBot
Fetches cricket articles from Wikipedia, splits them into chunks,
and stores them in a local ChromaDB vector database.

ChromaDB automatically embeds documents using the all-MiniLM-L6-v2
sentence-transformer model (downloaded on first run), so we don't
need to handle embeddings manually in Phase 1.

Run:  python ingest.py
"""

import requests
import chromadb

# ---------------------------------------------------------------
# 1. The knowledge base: Wikipedia articles to ingest.
#    Add or remove titles freely — anything on en.wikipedia.org.
# ---------------------------------------------------------------
ARTICLE_TITLES = [
    "Cricket",
    "Laws of Cricket",
    "Test cricket",
    "One Day International",
    "Twenty20",
    "Indian Premier League",
    "Cricket World Cup",
    "Duckworth–Lewis–Stern method",
    "Leg before wicket",
    "Glossary of cricket terms",
    "Sachin Tendulkar",
    "Virat Kohli",
    "MS Dhoni",
    "History of cricket",
    "Fielding (cricket)",
    "Bowling (cricket)",
    "Batting (cricket)",
    "The Ashes",
    "2023 Cricket World Cup",
    "ICC Men's T20 World Cup",
]

WIKI_API = "https://en.wikipedia.org/w/api.php"

# Wikipedia's API policy requires every client to identify itself with a
# User-Agent header. Requests without one get a 403 Forbidden error.
HEADERS = {
    "User-Agent": "CricBot/1.0 (educational RAG project; contact: student)"
}


def fetch_article(title: str) -> str:
    """Fetch the plain-text content of a Wikipedia article."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,   # plain text, no HTML
        "format": "json",
        "titles": title,
        "redirects": 1,
    }
    resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))  # there is exactly one page
    return page.get("extract", "")


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """
    Split text into overlapping chunks.

    Why chunk? Embedding models capture meaning best on short passages,
    and the LLM only needs the few most relevant passages — not whole
    articles. Overlap prevents losing context at chunk boundaries.
    """
    # Split on paragraphs first so we don't cut sentences awkwardly
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) <= chunk_size:
            current += " " + para
        else:
            if current.strip():
                chunks.append(current.strip())
            # start the next chunk with the tail of the previous one (overlap)
            current = current[-overlap:] + " " + para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def main():
    # PersistentClient saves the database to disk in ./cricbot_db
    client = chromadb.PersistentClient(path="cricbot_db")

    # Delete and recreate the collection so re-running ingest is clean
    try:
        client.delete_collection("cricket")
    except Exception:
        pass
    collection = client.create_collection("cricket")

    total_chunks = 0
    for title in ARTICLE_TITLES:
        print(f"Fetching: {title} ...", end=" ", flush=True)
        text = fetch_article(title)
        if not text:
            print("SKIPPED (no content found)")
            continue

        chunks = chunk_text(text)
        ids = [f"{title}-{i}" for i in range(len(chunks))]
        metadatas = [{"source": title} for _ in chunks]

        # ChromaDB embeds the documents automatically here
        collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        total_chunks += len(chunks)
        print(f"OK ({len(chunks)} chunks)")

    print(f"\nDone. Ingested {total_chunks} chunks from "
          f"{len(ARTICLE_TITLES)} articles into ./cricbot_db")


if __name__ == "__main__":
    main()
