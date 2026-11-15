"""Get extra tags from the Last.fm API (artist.getTopTags).

Needs a free API key: https://www.last.fm/api/account/create
    export LASTFM_API_KEY=your_key_here
    python src/collect_lastfm.py --limit 10     # quick test
    python src/collect_lastfm.py                # all artists

Outputs:
  data/interim/lastfm_raw.jsonl     one JSON line per artist (resumable)
  data/processed/lastfm_tags.csv    artistID, tag (cleaned), lastfm_count, lastfm_rank
"""
import argparse
import json
import os
import sys
import time

import pandas as pd
import requests

from clean import NON_DESCRIPTIVE_TAGS, OPINION_PATTERN, normalize_tag
from collect_deezer import similarity
from config import INTERIM_DIR, MATCH_THRESHOLD, PROCESSED_DIR

API = "https://ws.audioscrobbler.com/2.0/"
RAW_PATH = INTERIM_DIR / "lastfm_raw.jsonl"


def get_top_tags(name, key):
    params = {"method": "artist.gettoptags", "artist": name, "api_key": key,
              "format": "json", "autocorrect": 1}
    for attempt in range(4):
        try:
            r = requests.get(API, params=params, timeout=15)
            data = r.json()
        except (requests.RequestException, ValueError):
            time.sleep(2 * (attempt + 1))
            continue
        if "error" in data:
            if data["error"] == 29:          # rate limit
                time.sleep(5)
                continue
            return {"status": "error", "error": data.get("message")}
        tt = data.get("toptags", {})
        found = tt.get("@attr", {}).get("artist", name)
        return {"status": "ok", "lastfm_name": found,
                "match_score": round(similarity(name, found), 3),
                "tags": [(t["name"], int(t["count"])) for t in tt.get("tag", [])[:30]]}
    return {"status": "error", "error": "retries exhausted"}


def write_table():
    rows = [json.loads(l) for l in RAW_PATH.read_text().splitlines() if l.strip()]
    out = []
    for r in rows:
        if r.get("status") != "ok" or r.get("match_score", 0) < MATCH_THRESHOLD:
            continue
        seen = set()
        for raw, cnt in r["tags"]:
            t = normalize_tag(raw)
            if (len(t) < 2 or t in seen or t in NON_DESCRIPTIVE_TAGS
                    or pd.Series([t]).str.contains(OPINION_PATTERN, regex=True)[0] or cnt <= 0):
                continue
            seen.add(t)
            out.append({"artistID": r["artistID"], "tag": t, "lastfm_count": cnt})
    df = pd.DataFrame(out)
    if len(df):
        df["lastfm_rank"] = df.groupby("artistID").cumcount() + 1
    df.to_csv(PROCESSED_DIR / "lastfm_tags.csv", index=False)
    print(f"Saved {len(df):,} tag rows for {df['artistID'].nunique() if len(df) else 0:,} artists")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    key = os.environ.get("LASTFM_API_KEY")
    if not key:
        sys.exit("Set LASTFM_API_KEY first (free key: https://www.last.fm/api/account/create)")

    artists = pd.read_csv(PROCESSED_DIR / "artists_filtered.csv").sort_values(
        "n_listeners", ascending=False)
    if args.limit:
        artists = artists.head(args.limit)
    done = set()
    if RAW_PATH.exists():
        done = {json.loads(l)["artistID"] for l in RAW_PATH.read_text().splitlines() if l.strip()}
    todo = artists[~artists["artistID"].isin(done)]
    print(f"{len(done):,} already done, {len(todo):,} to go")

    with RAW_PATH.open("a") as fh:
        for i, r in enumerate(todo.itertuples(), 1):
            res = {"artistID": int(r.artistID), "name": r.name, **get_top_tags(r.name, key)}
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            time.sleep(0.25)                      # Last.fm asks for <= 5 requests/second
            if i % 100 == 0:
                print(f"  {i:,}/{len(todo):,}", flush=True)
    write_table()


if __name__ == "__main__":
    main()
