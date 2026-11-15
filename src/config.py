"""Shared paths and settings for the whole project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Data folders
RAW_DIR = ROOT / "data" / "raw" / "hetrec2011-lastfm-2k"
INTERIM_DIR = ROOT / "data" / "interim"        # API caches, half-finished files
PROCESSED_DIR = ROOT / "data" / "processed"    # final, clean tables
AUDIO_DIR = ROOT / "data" / "audio_previews"   # 30-second Deezer mp3 previews
FIG_DIR = ROOT / "reports" / "figures"

for d in (INTERIM_DIR, PROCESSED_DIR, AUDIO_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Cleaning rules
MIN_LISTENERS = 5          # drop artists with fewer distinct listeners
TAG_KS = (1, 3, 5)         # tag budgets tested in RQ2
MIN_TAG_USERS = 1          # a tag must be applied by at least this many users to an artist

# Name matching with Deezer / Last.fm
MATCH_THRESHOLD = 0.85     # similarity (0-1) needed to accept a search result

# Deezer allows 50 requests per 5 seconds; stay safely under it
DEEZER_MAX_REQ_PER_SEC = 8

# Audio features: 13 MFCC means + 13 MFCC stds + 12 chroma means = 38 numbers
N_MFCC = 13
SAMPLE_RATE = 22050

# Multi-song collection (CNN training data)
MAX_CLIPS_PER_ARTIST = 10
MELS_DIR = ROOT / "data" / "mels"          # mels/<artistID>/<trackID>.npy  (float16, n_mels x frames)
MEL_SR = 16000                             # spectrograms are made at 16 kHz to keep files small
N_MELS = 96
MEL_N_FFT = 1024
MEL_HOP = 512                              # 32 ms per frame -> 30 s clip ~ 938 frames
MELS_DIR.mkdir(parents=True, exist_ok=True)
