# From Tags to Sound — Explainable Cold-Start Recommendation for New Artists

**Question:** how can we recommend new artists who have no listeners yet?
A new artist has no listening history, so collaborative filtering cannot see them (cold start).
On upload day an artist *can* give **a few tags, a genre and audio**, so we describe artists with those.

| RQ | Question |
|----|----------|
| RQ1 | How much worse do new artists do than established ones (accuracy and exposure)? |
| RQ2 | Tags only vs audio only vs both? How many tags are needed: 1, 3 or 5? |
| RQ3 | Is the answer different for niche and mainstream genres? |

**Current stage: data collection, cleaning and EDA.** Modelling comes next.

---

## Folder structure

```
MusicProject/
├── data/
│   ├── raw/hetrec2011-lastfm-2k/   original HetRec 2011 files (never edited)
│   ├── interim/                    API caches (deezer_raw.jsonl, lastfm_raw.jsonl), logs
│   ├── audio_previews/             30-second Deezer mp3 previews, <artistID>.mp3
│   └── processed/                  clean tables used by the models
├── src/
│   ├── config.py                   paths and settings (thresholds, tag budgets, ...)
│   ├── load_data.py                read raw HetRec files
│   ├── clean.py                    filter artists, log1p, clean and rank tags
│   ├── collect_deezer.py           name match → genre → preview mp3 → 38 audio features
│   ├── collect_lastfm.py           extra tags from Last.fm (needs a free API key)
│   └── build_profiles.py           merge everything into artist_profiles.csv
├── notebooks/
│   ├── make_eda_notebook.py        generates the notebook
│   └── 01_eda.ipynb                exploratory data analysis (run, with outputs)
├── reports/
│   ├── figures/                    all EDA figures (PNG)
│   └── eda_summary_table.csv
├── models/                         (later) trained models
├── requirements.txt
└── README.md
```

## How to run (in this order)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 0. dataset (already downloaded into data/raw/)
#    https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip

python src/clean.py                 # 1. clean HetRec  (seconds)
python src/collect_deezer.py        # 2. Deezer genre + previews + audio features (resumable, ~1 h)
export LASTFM_API_KEY=...           # 3. optional: extra Last.fm tags
python src/collect_lastfm.py
python src/build_profiles.py        # 4. final artist profiles table
python src/collect_more_previews.py # 4b. up to 10 songs/artist -> mel-spectrograms + features (CNN data, ~1 h)
python notebooks/make_eda_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb   # 5. EDA
```

## Cleaning rules

| Step | Rule | Why |
|------|------|-----|
| Artist filter | keep artists with **≥ 5 distinct listeners** | removes 84% of artists, keeps 87% of plays |
| Play counts | `log_weight = log1p(weight)` | raw counts go from 1 to 352,698 |
| Tag text | lowercase, strip accents, `-`/`_`/`/` → space, `&` → `and`, synonyms (hiphop → hip hop) | merges spelling variants |
| Tag noise | drop opinion tags (seen live, awesome, favorites, <3, …) | they describe the listener, not the music |
| Tag ranking | per artist, by number of users who used the tag | the top 1 / 3 / 5 tags become the profile |
| Name match | normalised name similarity ≥ 0.85 (ties → most Deezer fans) | HetRec IDs are unknown to Deezer and Last.fm |
| Track choice | top track where the artist is the main artist (not "feat.") with a preview | one representative clip per artist |
| Genre | main genre of that track's album on Deezer | Deezer artists have no genre field |
| Audio | 13 MFCC means + 13 MFCC stds + 12 chroma means = **38 numbers** | timbre, timbre variation, pitch content |
| Scaling | audio saved **unscaled**; fit `StandardScaler` on the training split only | avoids data leakage |
| Mainstream / niche | genre with ≥ 5% of artists = mainstream, the rest niche | for RQ3 (provisional rule) |

## Output tables (`data/processed/`)

| File | One row per | Key columns |
|------|-------------|-------------|
| `interactions.csv` | user–artist pair | userID, artistID, weight, log_weight |
| `artists_filtered.csv` | kept artist | artistID, name, n_listeners, total_plays, popularity_bucket (head = top 20% by listeners) |
| `artist_tags.csv` | artist–tag | artistID, tag, n_users, rank |
| `artist_tags_topk.csv` | artist | tags_top1, tags_top3, tags_top5 (`|`-separated) |
| `deezer_artists.csv` | artist | deezer_id, deezer_name, match_score, status, track, album, genre |
| `audio_features.csv` | artist with audio | 38 features |
| **`artist_profiles.csv`** | artist | everything above merged: tags, genre, genre_group, audio |
| `clips.csv` | song (up to 10 per artist) | clip_id, artistID, track_id, title, album, pick_rank, mel_path |
| `clip_features.csv` | song | 38 audio features + tempo_bpm, rms_energy, spectral_centroid_hz, zero_crossing_rate |
| `artist_audio_features_multi.csv` | artist | mean of their songs' features, n_clips |

Spectrograms: `data/mels/<artistID>/<trackID>.npy`, log-mel, 96 bands × ~938 frames (16 kHz, hop 512), float16. The mp3s are deleted after processing.

**Cold start in one picture:** a new artist has **no rows in `interactions.csv`, only a row in `artist_profiles.csv`.**

## Key EDA results (full run, see `notebooks/01_eda.ipynb`)

| Fact | Value |
|------|-------|
| Raw | 1,892 users · 17,632 artists · 92,834 user–artist pairs · 99.72% sparse |
| Long tail | 60.6% of artists have 1 listener; top 1% of artists = 52% of plays |
| After ≥5 filter | 1,881 users · 2,828 artists · 71,426 interactions (84.0% of artists removed, 87.1% of plays kept) |
| Tags | 9,749 raw → 6,392 clean tags; 99.6% of artists have ≥1 tag, 90.7% have 5 |
| **Identical tag profiles** | artists sharing their exact top-k tag set: **k=1 97.8% · k=3 58.2% · k=5 22.4%** → why audio is needed |
| Deezer match | 97.8% exact name match, 98.5% accepted |
| Genre / audio | 93.7% have a genre · 95.9% have audio (2,713 previews) · 93.4% complete profiles |
| Genre groups | 2,022 mainstream (Rock, Pop, Alternative, Electro) / 628 niche |
| Audio scales | feature std from 0.11 to 84 → scaling required; 22 PCA components hold 90% variance |
| Multi-song audio (CNN data) | 26,015 songs for 2,708 artists · 9.6 songs and 8.5 albums per artist · 92% have all 10 · ~260k three-second pieces · 4.4 GB of mel-spectrograms |
| RQ3 hint | genres that separate best in audio space: Country, Metal (niche); mainstream genres overlap more |

## Known limitations
- Deezer's top track reflects today's popularity, not 2011.
- Genre comes from one album, so crossover artists may get the wrong genre.
- Fuzzy name matching can still pick the wrong artist for short or common names; see `low_match` rows in `deezer_artists.csv`.
- One 30-second clip per artist may not represent the whole catalogue.
