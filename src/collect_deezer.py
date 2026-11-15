"""Collect Deezer genre + 30-second audio preview for every kept artist, and turn
each preview into 38 audio numbers with librosa.

Steps per artist (HetRec gives only a name, Deezer does not know HetRec IDs):
  1. search the name on Deezer         -> candidate artists
  2. check the match (name similarity) -> accept or reject
  3. artist's top tracks               -> first track that has a preview mp3
  4. that track's album                -> genre
  5. download preview, librosa         -> 38 features

Results are appended to data/interim/deezer_raw.jsonl, so the script can be stopped
and restarted: artists already done are skipped.

Outputs:
  data/interim/deezer_raw.jsonl       one JSON line per artist (raw result)
  data/processed/deezer_artists.csv   match info + genre + preview info
  data/processed/audio_features.csv   artistID + 38 audio features
  data/audio_previews/<artistID>.mp3  the previews (unless --no-keep-audio)

Run:
  python src/collect_deezer.py --limit 10      # quick test
  python src/collect_deezer.py                 # all artists (~30-60 min)
"""
import argparse
import json
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import requests

from config import (AUDIO_DIR, DEEZER_MAX_REQ_PER_SEC, INTERIM_DIR, MATCH_THRESHOLD,
                    N_MFCC, PROCESSED_DIR, SAMPLE_RATE)

API = "https://api.deezer.com"
RAW_PATH = INTERIM_DIR / "deezer_raw.jsonl"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "music-coldstart-research/0.1"


# ---------------------------------------------------------------- rate limit
class RateLimiter:
    def __init__(self, per_sec):
        self.interval = 1.0 / per_sec
        self.lock = threading.Lock()
        self.next_t = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next_t:
                time.sleep(self.next_t - now)
            self.next_t = max(now, self.next_t) + self.interval


LIMIT = RateLimiter(DEEZER_MAX_REQ_PER_SEC)


def api_get(path, params=None, retries=4):
    for attempt in range(retries):
        LIMIT.wait()
        try:
            r = SESSION.get(f"{API}{path}", params=params, timeout=15)
            data = r.json()
        except (requests.RequestException, ValueError):
            time.sleep(2 * (attempt + 1))
            continue
        if isinstance(data, dict) and "error" in data:
            if data["error"].get("code") == 4:      # quota exceeded -> back off
                time.sleep(5)
                continue
            return None
        return data
    return None


# ---------------------------------------------------------------- matching
def norm_name(s):
    # strip accents but keep non-Latin letters (Cyrillic, CJK, ...)
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    s = s.replace("&", " and ").replace("$", "s")
    s = re.sub(r"^the\s+", "", s)
    s = re.sub(r"[\W_]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_junk_name(name):
    """Placeholder / URL / symbol-only names in HetRec that are not real artists."""
    n = str(name).strip().lower()
    return (n in {"[unknown]", "unknown", "various artists", "va"}
            or n.startswith(("www.", "http")) or not re.search(r"\w", n))


def similarity(a, b):
    a, b = norm_name(a), norm_name(b)
    if not a or not b:
        return 0.0
    return 1.0 if a == b else SequenceMatcher(None, a, b).ratio()


def match_artist(name):
    res = api_get("/search/artist", {"q": name, "limit": 10})
    if not res or not res.get("data"):
        return None, 0.0
    scored = [(similarity(name, c["name"]), c.get("nb_fan", 0) or 0, c) for c in res["data"]]
    # 1. an exact (normalised) name match always wins; ties -> most fans
    exact = [x for x in scored if x[0] == 1.0]
    if exact:
        best = max(exact, key=lambda x: x[1])
        return best[2], 1.0
    # 2. otherwise accept fuzzy matches (>= threshold) and "containment" matches, e.g.
    #    "Bob Marley" ~ "Bob Marley & The Wailers", "The Jimi Hendrix Experience" ~ "Jimi Hendrix".
    #    Among those, the one with the most fans is the real artist (not "Bob Marley Jr" or a tribute act).
    #    Containment = one name is the START of the other, >= 2 words, every word >= 2 letters
    #    (so "Godspeed You! Black Emperor" does NOT match "Black Emperor").
    q = norm_name(name).split()
    ok = []
    for sim, fans, c in scored:
        t = norm_name(c["name"]).split()
        short, long_ = (q, t) if len(q) <= len(t) else (t, q)
        contained = (len(short) >= 2 and all(len(w) >= 2 for w in short)
                     and long_[:len(short)] == short)
        if sim >= MATCH_THRESHOLD or contained:
            ok.append((max(sim, 0.9 if contained else 0), fans, c))
    if ok:
        best = max(ok, key=lambda x: x[1])
        return best[2], best[0]
    best = max(scored, key=lambda x: (x[0], x[1]))
    return best[2], best[0]


# ---------------------------------------------------------------- audio
FEATURE_NAMES = ([f"mfcc{i+1}_mean" for i in range(N_MFCC)]
                 + [f"mfcc{i+1}_std" for i in range(N_MFCC)]
                 + [f"chroma{i+1}_mean" for i in range(12)])


def audio_features(path):
    import librosa  # imported here so the module loads fast
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(y) < sr:  # shorter than 1 second -> unusable
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    vec = np.concatenate([mfcc.mean(1), mfcc.std(1), chroma.mean(1)])
    return dict(zip(FEATURE_NAMES, vec.astype(float).round(6)))


# ---------------------------------------------------------------- one artist
def process_artist(artist_id, name, keep_audio=True):
    out = {"artistID": int(artist_id), "name": name, "status": "ok"}
    if is_junk_name(name):
        out.update(status="invalid_name", match_score=0.0)
        return out
    cand, score = match_artist(name)
    out["match_score"] = round(score, 3)
    if cand is None:
        out["status"] = "not_found"
        return out
    out.update(deezer_id=cand["id"], deezer_name=cand["name"], deezer_fans=cand.get("nb_fan"))
    if score < MATCH_THRESHOLD:
        out["status"] = "low_match"
        return out

    top = api_get(f"/artist/{cand['id']}/top", {"limit": 10})
    tracks = [t for t in (top or {}).get("data", []) if t.get("preview")]
    # prefer songs where this artist is the main artist and not a "feat." guest
    own = [t for t in tracks if t.get("artist", {}).get("id") == cand["id"]
           and not re.search(r"\b(feat|ft)\b", t.get("title", ""), re.I)]
    tracks = own or tracks
    if not tracks:
        out["status"] = "no_preview"
        return out
    t = tracks[0]
    out.update(track_id=t["id"], track_title=t.get("title"),
               album_id=t.get("album", {}).get("id"), album_title=t.get("album", {}).get("title"))

    album = api_get(f"/album/{out['album_id']}") if out["album_id"] else None
    genres = [g["name"] for g in (album or {}).get("genres", {}).get("data", [])]
    out["genre"] = genres[0] if genres else None
    out["all_genres"] = "|".join(genres)

    mp3 = AUDIO_DIR / f"{artist_id}.mp3"
    try:
        if not mp3.exists():
            r = SESSION.get(t["preview"], timeout=30)
            r.raise_for_status()
            mp3.write_bytes(r.content)
        feats = audio_features(str(mp3))
    except Exception as e:  # noqa: BLE001 - record any download/decoding failure
        out["status"] = "audio_error"
        out["error"] = str(e)[:200]
        feats = None
    finally:
        if not keep_audio and mp3.exists():
            mp3.unlink()
    if feats is None and out["status"] == "ok":
        out["status"] = "audio_error"
    if feats:
        out["features"] = feats
    return out


# ---------------------------------------------------------------- outputs
def write_tables():
    rows = [json.loads(l) for l in RAW_PATH.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows).drop_duplicates("artistID", keep="last")
    feats = df.dropna(subset=["features"]) if "features" in df else df.iloc[0:0]
    meta = df.drop(columns=[c for c in ("features",) if c in df])
    meta.to_csv(PROCESSED_DIR / "deezer_artists.csv", index=False)
    if len(feats):
        f = pd.DataFrame(list(feats["features"]))
        f.insert(0, "artistID", feats["artistID"].values)
        f.to_csv(PROCESSED_DIR / "audio_features.csv", index=False)
    print("\nStatus counts:\n", meta["status"].value_counts().to_string())
    print(f"Artists with audio features: {len(feats):,} / {len(meta):,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N artists")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-keep-audio", action="store_true", help="delete mp3 after features")
    args = ap.parse_args()

    artists = pd.read_csv(PROCESSED_DIR / "artists_filtered.csv")
    artists = artists.sort_values("n_listeners", ascending=False)
    if args.limit:
        artists = artists.head(args.limit)

    done = set()
    if RAW_PATH.exists():
        done = {json.loads(l)["artistID"] for l in RAW_PATH.read_text().splitlines() if l.strip()}
    todo = artists[~artists["artistID"].isin(done)]
    print(f"{len(done):,} already done, {len(todo):,} to go")

    lock = threading.Lock()
    with RAW_PATH.open("a") as fh, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process_artist, r.artistID, r.name, not args.no_keep_audio)
                for r in todo.itertuples()]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            with lock:
                fh.write(json.dumps(res) + "\n")
                fh.flush()
            if i % 50 == 0 or i == len(futs):
                print(f"  {i:,}/{len(futs):,} done", flush=True)

    write_tables()


if __name__ == "__main__":
    main()
