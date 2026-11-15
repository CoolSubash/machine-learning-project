"""Collect up to 10 songs per artist from Deezer (CNN training data).

One artist can make very different songs, so one preview is not enough. For every
artist that was matched in collect_deezer.py, this script:
  1. gets the artist's top 50 tracks on Deezer,
  2. keeps songs where the artist is the main artist (not "feat."), with a preview,
     and skips duplicates (same title, live, remix, karaoke, instrumental versions),
  3. picks up to 10, spread over different albums (round-robin by album),
  4. for each song: downloads the 30-second preview, saves a log-mel spectrogram,
     computes 38 + 4 audio features, then DELETES the mp3 (only features are kept).

Outputs:
  data/mels/<artistID>/<trackID>.npy          log-mel spectrogram, float16, 96 x ~938
  data/interim/clips_raw.jsonl                 one JSON line per artist (resumable)
  data/processed/clips.csv                     one row per song (metadata + mel path)
  data/processed/clip_features.csv            one row per song: 38 features + tempo, rms, centroid, zcr
  data/processed/artist_audio_features_multi.csv   artist = mean over their songs

Run:
  python src/collect_more_previews.py --limit 5     # quick test
  python src/collect_more_previews.py               # all artists (~1-2 h, resumable)
"""
import argparse
import json
import os
import re
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from config import (INTERIM_DIR, MAX_CLIPS_PER_ARTIST, MEL_HOP, MEL_N_FFT, MEL_SR, MELS_DIR,
                    N_MELS, N_MFCC, PROCESSED_DIR, SAMPLE_RATE)

API = "https://api.deezer.com"
RAW_PATH = INTERIM_DIR / "clips_raw.jsonl"
SKIP_VERSION = re.compile(r"\b(live|remix|remixed|karaoke|instrumental|demo|re-?recorded|"
                          r"sped up|slowed|8-?bit|cover)\b", re.I)
FEAT = re.compile(r"\b(feat|ft)\b\.?", re.I)

FEATURE_NAMES = ([f"mfcc{i+1}_mean" for i in range(N_MFCC)]
                 + [f"mfcc{i+1}_std" for i in range(N_MFCC)]
                 + [f"chroma{i+1}_mean" for i in range(12)])
EXPLAIN_NAMES = ["tempo_bpm", "rms_energy", "spectral_centroid_hz", "zero_crossing_rate"]


def api_get(session, path, params=None, retries=4):
    for attempt in range(retries):
        try:
            data = session.get(f"{API}{path}", params=params, timeout=15).json()
        except (requests.RequestException, ValueError):
            time.sleep(2 * (attempt + 1))
            continue
        if isinstance(data, dict) and "error" in data:
            if data["error"].get("code") == 4:      # quota -> wait and retry
                time.sleep(5)
                continue
            return None
        return data
    return None


def base_title(title):
    """'Creep (Acoustic)' / 'Creep - Remastered 2009' -> 'creep'."""
    t = re.sub(r"[\(\[].*?[\)\]]", " ", title.lower())
    t = re.split(r"\s+-\s+", t)[0]
    return re.sub(r"[^\w]+", " ", t).strip()


def choose_tracks(tracks, deezer_id, k):
    own = [t for t in tracks if t.get("preview") and t.get("artist", {}).get("id") == deezer_id
           and not FEAT.search(t.get("title", ""))]
    pool = own or [t for t in tracks if t.get("preview")]
    seen, clean = set(), []
    for t in pool:                                   # already in popularity order
        full = t.get("title", "") + " " + t.get("title_version", "")
        b = base_title(t.get("title", ""))
        if not b or b in seen or SKIP_VERSION.search(full):
            continue
        seen.add(b)
        clean.append(t)
    # round-robin over albums: best song of each album first, then second-best, ...
    by_album = {}
    for t in clean:
        by_album.setdefault(t.get("album", {}).get("id"), []).append(t)
    chosen, rnd = [], 0
    while len(chosen) < k and any(len(v) > rnd for v in by_album.values()):
        for v in by_album.values():
            if len(v) > rnd and len(chosen) < k:
                chosen.append(v[rnd])
        rnd += 1
    return chosen


def process_clip(path):
    import librosa
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(y) < 5 * sr:
        return None, None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    vec = np.concatenate([mfcc.mean(1), mfcc.std(1), chroma.mean(1)])
    feats = dict(zip(FEATURE_NAMES, vec.astype(float).round(6)))
    tempo = librosa.feature.tempo(y=y, sr=sr)
    feats.update({
        "tempo_bpm": round(float(np.atleast_1d(tempo)[0]), 2),
        "rms_energy": round(float(librosa.feature.rms(y=y).mean()), 6),
        "spectral_centroid_hz": round(float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()), 2),
        "zero_crossing_rate": round(float(librosa.feature.zero_crossing_rate(y).mean()), 6),
    })
    y16 = librosa.resample(y, orig_sr=sr, target_sr=MEL_SR)
    mel = librosa.feature.melspectrogram(y=y16, sr=MEL_SR, n_fft=MEL_N_FFT, hop_length=MEL_HOP,
                                         n_mels=N_MELS)
    logmel = librosa.power_to_db(mel, ref=1.0).astype(np.float16)   # absolute dB, not per-clip max
    return feats, logmel


def process_artist(artist_id, deezer_id, k):
    s = requests.Session()
    s.headers["User-Agent"] = "music-coldstart-research/0.1"
    top = api_get(s, f"/artist/{int(deezer_id)}/top", {"limit": 50}) or {}
    chosen = choose_tracks(top.get("data", []), int(deezer_id), k)
    out = {"artistID": int(artist_id), "deezer_id": int(deezer_id),
           "n_candidates": len(top.get("data", [])), "clips": []}
    adir = MELS_DIR / str(artist_id)
    for rank, t in enumerate(chosen, 1):
        clip = {"clip_id": f"{artist_id}_{t['id']}", "track_id": t["id"], "title": t.get("title"),
                "album_id": t.get("album", {}).get("id"), "album_title": t.get("album", {}).get("title"),
                "pick_rank": rank, "deezer_rank": t.get("rank"), "status": "ok"}
        mel_path = adir / f"{t['id']}.npy"
        try:
            r = s.get(t["preview"], timeout=30)
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(r.content)
                tmp = f.name
            try:
                feats, logmel = process_clip(tmp)
            finally:
                os.unlink(tmp)                         # the mp3 is not kept
            if feats is None:
                clip["status"] = "too_short"
            else:
                adir.mkdir(parents=True, exist_ok=True)
                np.save(mel_path, logmel)
                clip["mel_path"] = str(mel_path.relative_to(MELS_DIR.parent.parent))
                clip["n_frames"] = int(logmel.shape[1])
                clip["features"] = feats
        except Exception as e:  # noqa: BLE001 - record any download/decoding failure
            clip["status"] = "error"
            clip["error"] = str(e)[:200]
        out["clips"].append(clip)
    return out


def write_tables():
    rows = [json.loads(l) for l in RAW_PATH.read_text().splitlines() if l.strip()]
    rows = list({r["artistID"]: r for r in rows}.values())
    clips, feats = [], []
    for r in rows:
        for c in r["clips"]:
            meta = {k: v for k, v in c.items() if k != "features"}
            clips.append({"artistID": r["artistID"], **meta})
            if c.get("features"):
                feats.append({"clip_id": c["clip_id"], "artistID": r["artistID"], **c["features"]})
    clips, feats = pd.DataFrame(clips), pd.DataFrame(feats)
    clips.to_csv(PROCESSED_DIR / "clips.csv", index=False)
    feats.to_csv(PROCESSED_DIR / "clip_features.csv", index=False)
    art = feats.drop(columns="clip_id").groupby("artistID").mean().round(6)
    art.insert(0, "n_clips", feats.groupby("artistID").size())
    art.reset_index().to_csv(PROCESSED_DIR / "artist_audio_features_multi.csv", index=False)

    per_artist = feats.groupby("artistID").size()
    print(f"\nArtists processed: {len(rows):,}")
    print(f"Songs with spectrogram + features: {len(feats):,}")
    print(f"Artists with >=1 song: {len(per_artist):,}   mean songs/artist: {per_artist.mean():.1f}")
    print("Songs per artist:", per_artist.value_counts().sort_index().to_dict())
    print("Clip status:", clips["status"].value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--k", type=int, default=MAX_CLIPS_PER_ARTIST)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    dz = pd.read_csv(PROCESSED_DIR / "deezer_artists.csv")
    dz = dz[dz["status"].isin(["ok", "no_preview", "audio_error"]) & dz["deezer_id"].notna()]
    art = pd.read_csv(PROCESSED_DIR / "artists_filtered.csv")[["artistID", "n_listeners"]]
    dz = dz.merge(art, on="artistID").sort_values("n_listeners", ascending=False)
    if args.limit:
        dz = dz.head(args.limit)

    done = set()
    if RAW_PATH.exists():
        done = {json.loads(l)["artistID"] for l in RAW_PATH.read_text().splitlines() if l.strip()}
    todo = dz[~dz["artistID"].isin(done)]
    print(f"{len(done):,} artists already done, {len(todo):,} to go, up to {args.k} songs each, "
          f"{args.workers} workers", flush=True)

    t0 = time.time()
    with RAW_PATH.open("a") as fh, ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process_artist, r.artistID, r.deezer_id, args.k) for r in todo.itertuples()]
        n_clips = 0
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            n_clips += sum(c["status"] == "ok" for c in res["clips"])
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            if i % 50 == 0 or i == len(futs):
                el = time.time() - t0
                print(f"  {i:,}/{len(futs):,} artists, {n_clips:,} songs, "
                      f"{el/60:.1f} min, ~{el/i*(len(futs)-i)/60:.0f} min left", flush=True)
    write_tables()


if __name__ == "__main__":
    main()
