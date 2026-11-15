"""Merge everything into the final artist profiles table.

A *profile* is what a brand-new artist could give on upload day:
tags (top 1 / 3 / 5), a genre, and 38 audio numbers. No listening data.

Output: data/processed/artist_profiles.csv
  artistID, name, n_listeners, total_plays, popularity_bucket,
  tags_top1, tags_top3, tags_top5, n_clean_tags, tag_source,
  deezer_id, deezer_name, match_score, deezer_status, genre, genre_group,
  has_audio, mfcc1_mean ... chroma12_mean

NOTE: audio features are saved UNSCALED on purpose. Scaling (StandardScaler) must be
fitted on the training split only, later, to avoid leaking test information.

Run: python src/build_profiles.py
"""
import pandas as pd

from config import PROCESSED_DIR, TAG_KS

# Genres holding at least this share of matched artists count as "mainstream" (RQ3).
MAINSTREAM_MIN_SHARE = 0.05


def merged_tags():
    """HetRec tags first; Last.fm tags (if collected) fill the empty slots up to 5."""
    het = pd.read_csv(PROCESSED_DIR / "artist_tags.csv")[["artistID", "tag", "rank"]]
    het["source"] = "hetrec"
    lfm_path = PROCESSED_DIR / "lastfm_tags.csv"
    if lfm_path.exists() and lfm_path.stat().st_size > 1:
        lfm = pd.read_csv(lfm_path).rename(columns={"lastfm_rank": "rank"})
        lfm = lfm[["artistID", "tag", "rank"]].assign(source="lastfm", rank=lambda d: d["rank"] + 1000)
        het = pd.concat([het, lfm]).drop_duplicates(["artistID", "tag"], keep="first")
    het = het.sort_values(["artistID", "rank"])
    het["rank"] = het.groupby("artistID").cumcount() + 1
    k_max = max(TAG_KS)
    top = het[het["rank"] <= k_max]
    out = top.groupby("artistID").agg(
        n_clean_tags=("tag", "size"),
        tag_source=("source", lambda s: "+".join(sorted(set(s)))))
    for k in TAG_KS:
        out[f"tags_top{k}"] = top[top["rank"] <= k].groupby("artistID")["tag"].apply("|".join)
    return out.reset_index()


def main():
    art = pd.read_csv(PROCESSED_DIR / "artists_filtered.csv").drop(columns=["has_tags"])
    prof = art.merge(merged_tags(), on="artistID", how="left")
    prof["n_clean_tags"] = prof["n_clean_tags"].fillna(0).astype(int)

    dz = pd.read_csv(PROCESSED_DIR / "deezer_artists.csv")
    dz = dz[["artistID", "deezer_id", "deezer_name", "match_score", "status", "genre"]]
    prof = prof.merge(dz.rename(columns={"status": "deezer_status"}), on="artistID", how="left")
    ok = prof["deezer_status"].isin(["ok", "audio_error", "no_preview"])
    prof.loc[~ok, "genre"] = None    # don't trust the genre of a rejected match

    share = prof["genre"].value_counts(normalize=True)
    mainstream = set(share[share >= MAINSTREAM_MIN_SHARE].index)
    prof["genre_group"] = prof["genre"].map(
        lambda g: None if pd.isna(g) else ("mainstream" if g in mainstream else "niche"))

    af = pd.read_csv(PROCESSED_DIR / "audio_features.csv")
    prof = prof.merge(af, on="artistID", how="left")
    prof["has_audio"] = prof["mfcc1_mean"].notna()

    prof.to_csv(PROCESSED_DIR / "artist_profiles.csv", index=False)
    print(f"Profiles: {len(prof):,} artists")
    print(f"  with >=1 tag : {(prof['n_clean_tags'] > 0).mean():.1%}")
    print(f"  with 5 tags  : {(prof['n_clean_tags'] >= 5).mean():.1%}")
    print(f"  with genre   : {prof['genre'].notna().mean():.1%}")
    print(f"  with audio   : {prof['has_audio'].mean():.1%}")
    print(f"  complete (tags+genre+audio): "
          f"{((prof['n_clean_tags'] > 0) & prof['genre'].notna() & prof['has_audio']).mean():.1%}")
    print(f"  mainstream genres: {sorted(mainstream)}")


if __name__ == "__main__":
    main()
