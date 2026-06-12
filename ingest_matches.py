"""
ingest_matches.py — Phase 2 of CricBot
Downloads ball-by-ball IPL match data from Cricsheet.org, converts each
match's JSON into a natural-language summary, and ADDS those summaries to
the existing ChromaDB collection (alongside the Wikipedia chunks).

Why convert JSON to sentences? Embedding models understand language, not
nested data structures. "Mumbai Indians beat Chennai Super Kings by 5
wickets at Wankhede Stadium" embeds meaningfully; raw JSON does not.

Run:  python ingest_matches.py
"""

import io
import json
import zipfile
from collections import defaultdict

import requests
import chromadb

# Cricsheet provides free, open ball-by-ball data for thousands of matches.
# ipl_json.zip = every IPL match ever played, one JSON file per match.
CRICSHEET_URL = "https://cricsheet.org/downloads/ipl_json.zip"

HEADERS = {
    "User-Agent": "CricBot/1.0 (educational RAG project; contact: student)"
}


def download_matches() -> dict[str, dict]:
    """Download the Cricsheet zip and return {filename: parsed_json}."""
    print("Downloading IPL data from Cricsheet (~30 MB)...")
    resp = requests.get(CRICSHEET_URL, headers=HEADERS, timeout=120)
    resp.raise_for_status()

    matches = {}
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        print(f"Found {len(names)} match files. Parsing...")
        for name in names:
            with zf.open(name) as f:
                matches[name] = json.load(f)
    return matches


def summarize_match(match: dict) -> str | None:
    """Turn one Cricsheet match JSON into a natural-language summary."""
    info = match.get("info", {})
    teams = info.get("teams", [])
    if len(teams) != 2:
        return None

    date = info.get("dates", ["unknown date"])[0]
    season = info.get("season", "")
    venue = info.get("venue", "an unknown venue")
    city = info.get("city", "")
    event = info.get("event", {})
    stage = event.get("stage") or (
        f"match {event.get('match_number')}" if event.get("match_number") else ""
    )

    # --- Result ---
    outcome = info.get("outcome", {})
    if "winner" in outcome:
        by = outcome.get("by", {})
        if "runs" in by:
            result = f"{outcome['winner']} won by {by['runs']} runs"
        elif "wickets" in by:
            result = f"{outcome['winner']} won by {by['wickets']} wickets"
        else:
            result = f"{outcome['winner']} won"
        if outcome.get("method"):
            result += f" ({outcome['method']} method)"
    elif outcome.get("result") == "tie":
        winner = outcome.get("eliminator") or outcome.get("bowl_out")
        result = (f"the match was tied, with {winner} winning the tie-breaker"
                  if winner else "the match was tied")
    else:
        result = "the match had no result"

    # --- Toss ---
    toss = info.get("toss", {})
    toss_text = ""
    if toss:
        toss_text = (f" {toss.get('winner')} won the toss and chose to "
                     f"{toss.get('decision')}.")

    # --- Innings: totals, top batters, top bowlers ---
    innings_lines = []
    for inn in match.get("innings", []):
        team = inn.get("team", "Unknown team")
        runs_total = 0
        wickets = 0
        batter_runs = defaultdict(int)
        bowler_wkts = defaultdict(int)

        for over in inn.get("overs", []):
            for d in over.get("deliveries", []):
                runs_total += d.get("runs", {}).get("total", 0)
                batter_runs[d.get("batter", "?")] += d.get("runs", {}).get("batter", 0)
                for w in d.get("wickets", []):
                    wickets += 1
                    # run outs aren't credited to the bowler
                    if w.get("kind") not in ("run out", "retired hurt",
                                             "retired out", "obstructing the field"):
                        bowler_wkts[d.get("bowler", "?")] += 1

        top_bats = sorted(batter_runs.items(), key=lambda x: -x[1])[:2]
        bat_text = ", ".join(f"{n} scored {r}" for n, r in top_bats if r > 0)

        top_bowls = sorted(bowler_wkts.items(), key=lambda x: -x[1])[:2]
        bowl_text = ", ".join(
            f"{n} took {w} wicket{'s' if w != 1 else ''}"
            for n, w in top_bowls if w > 0
        )

        line = f"{team} made {runs_total}/{wickets}"
        if bat_text:
            line += f" ({bat_text})"
        if bowl_text:
            line += f"; for the bowling side, {bowl_text}"
        innings_lines.append(line + ".")

    pom = info.get("player_of_match", [])
    pom_text = f" Player of the match: {pom[0]}." if pom else ""

    where = f"{venue}, {city}" if city and city not in venue else venue
    header = (f"IPL {season}{', ' + stage if stage else ''}: "
              f"{teams[0]} vs {teams[1]} at {where} on {date}.")

    return (f"{header}{toss_text} {result[0].upper() + result[1:]}. "
            + " ".join(innings_lines) + pom_text)


def main():
    matches = download_matches()

    client = chromadb.PersistentClient(path="cricbot_db")
    # get_or_create: we ADD to the existing collection, keeping Wikipedia chunks
    collection = client.get_or_create_collection("cricket")

    docs, ids, metas = [], [], []
    skipped = 0
    for name, match in matches.items():
        summary = summarize_match(match)
        if not summary:
            skipped += 1
            continue
        info = match.get("info", {})
        teams = info.get("teams", ["?", "?"])
        season = info.get("season", "?")
        docs.append(summary)
        ids.append(f"match-{name}")
        metas.append({"source": f"IPL {season}: {teams[0]} vs {teams[1]}"})

    print(f"Built {len(docs)} match summaries ({skipped} skipped).")
    print("Embedding and storing (this takes a few minutes)...")

    # Add in batches — large single adds can hit limits
    BATCH = 200
    for i in range(0, len(docs), BATCH):
        collection.add(
            documents=docs[i:i + BATCH],
            ids=ids[i:i + BATCH],
            metadatas=metas[i:i + BATCH],
        )
        print(f"  stored {min(i + BATCH, len(docs))}/{len(docs)}")

    print(f"\nDone. Collection now holds {collection.count()} total chunks "
          f"(Wikipedia + match summaries).")


if __name__ == "__main__":
    main()
