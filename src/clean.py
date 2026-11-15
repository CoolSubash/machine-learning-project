"""Clean the HetRec data and write the processed tables.

Outputs (data/processed/):
  interactions.csv      userID, artistID, weight, log_weight     (who listened to whom)
  artists_filtered.csv  artistID, name, n_listeners, total_plays, popularity_bucket
  artist_tags.csv       artistID, tag, n_users, rank            (cleaned tags, ranked per artist)
  artist_tags_topk.csv  artistID, tags_top1, tags_top3, tags_top5

Run:  python src/clean.py
"""
import re
import unicodedata

import numpy as np
import pandas as pd

from config import MIN_LISTENERS, PROCESSED_DIR, TAG_KS
from load_data import load_artists, load_tags, load_user_artists, load_user_tagged_artists

# ---------------------------------------------------------------- tag cleaning
# Tags that say what the *user* thinks or did, not what the music sounds like.
NON_DESCRIPTIVE_TAGS = {
    "seen live", "awesome", "amazing", "love", "loved", "cool", "favorites", "favourites",
    "favorite", "favourite", "favorite songs", "favourite songs", "favorite artists",
    "favourite artists", "my favorite", "my favourite", "love at first listen", "<3",
    "best", "good", "great", "perfect", "the best", "genius", "check out", "to listen",
    "albums i own", "owned", "spotify", "under 2000 listeners", "all", "music",
    "cover", "covers", "remix", "remixes", "favorite bands", "favourite bands",
}

# Any tag containing these words is an opinion ("fucking awesome", "my favourite band")
OPINION_PATTERN = (r"\b(?:awesome|amazing|favou?rites?|fav|best|love it|loved|fucking|"
                   r"seen live|i own|check out|<3)\b")

# Spelling variants that mean the same thing -> one canonical form
TAG_SYNONYMS = {
    "hiphop": "hip hop",
    "rnb": "r and b", "r b": "r and b",
    "synthpop": "synth pop",
    "female vocalist": "female vocalists",
    "male vocalist": "male vocalists",
    "brasil": "brazilian",
    "electronica": "electronica",
    "post punk": "post punk",
    "chill out": "chillout",
    "lofi": "lo fi",
    "triphop": "trip hop",
    "drum n bass": "drum and bass", "dnb": "drum and bass", "d and b": "drum and bass",
}


def normalize_tag(tag: str) -> str:
    """Lowercase, remove accents, unify separators, apply synonyms."""
    t = unicodedata.normalize("NFKD", str(tag)).encode("ascii", "ignore").decode()
    t = t.lower().strip()
    t = t.replace("&", " and ")
    t = re.sub(r"[-_/.]+", " ", t)           # hip-hop, hip_hop -> hip hop
    t = re.sub(r"[^a-z0-9<3 ']+", " ", t)     # drop odd punctuation
    t = re.sub(r"\s+", " ", t).strip()
    return TAG_SYNONYMS.get(t, t)


def clean_tags(user_tagged: pd.DataFrame, tags: pd.DataFrame, keep_artists) -> pd.DataFrame:
    """Return one row per (artist, clean tag) with the number of users who used it, ranked."""
    df = user_tagged.merge(tags, on="tagID", how="left")
    df = df[df["artistID"].isin(keep_artists)].copy()
    df["tag"] = df["tagValue"].map(normalize_tag)
    df = df[(df["tag"].str.len() > 1)
            & (~df["tag"].isin(NON_DESCRIPTIVE_TAGS))
            & (~df["tag"].str.contains(OPINION_PATTERN, regex=True))]

    per_artist = (df.groupby(["artistID", "tag"])["userID"].nunique()
                    .rename("n_users").reset_index())
    # tie-breaker: how many artists in the whole dataset carry this tag (more common = more general)
    tag_df = per_artist.groupby("tag")["artistID"].nunique().rename("tag_df")
    per_artist = per_artist.merge(tag_df, on="tag")
    per_artist = per_artist.sort_values(["artistID", "n_users", "tag_df", "tag"],
                                        ascending=[True, False, False, True])
    per_artist["rank"] = per_artist.groupby("artistID").cumcount() + 1
    return per_artist.drop(columns="tag_df")


def top_k_tags(artist_tags: pd.DataFrame) -> pd.DataFrame:
    out = None
    for k in TAG_KS:
        col = (artist_tags[artist_tags["rank"] <= k]
               .groupby("artistID")["tag"].apply(lambda s: "|".join(s))
               .rename(f"tags_top{k}"))
        out = col.to_frame() if out is None else out.join(col, how="outer")
    return out.reset_index()


# ---------------------------------------------------------------- main
def main():
    artists = load_artists()
    ua = load_user_artists()
    tags = load_tags()
    uta = load_user_tagged_artists()

    # 1. basic checks
    assert not ua.duplicated(["userID", "artistID"]).any(), "duplicate user-artist pairs"
    assert (ua["weight"] > 0).all(), "non-positive play counts"

    # 2. filter rare artists
    listeners = ua.groupby("artistID")["userID"].nunique().rename("n_listeners")
    plays = ua.groupby("artistID")["weight"].sum().rename("total_plays")
    keep = listeners[listeners >= MIN_LISTENERS].index
    print(f"Artists with plays: {len(listeners):,}  ->  kept (>= {MIN_LISTENERS} listeners): "
          f"{len(keep):,}  ({1 - len(keep) / len(listeners):.1%} removed)")
    print(f"Plays kept: {plays.loc[keep].sum() / plays.sum():.1%}")

    inter = ua[ua["artistID"].isin(keep)].copy()
    inter["log_weight"] = np.log1p(inter["weight"])   # shrink 1..352,698 to 0.7..12.8

    # 3. artist table
    art = (artists[["artistID", "name"]]
           .merge(listeners, on="artistID").merge(plays, on="artistID"))
    art = art[art["artistID"].isin(keep)].copy()
    art["name"] = art["name"].str.strip()
    # popularity buckets by listener count (top 20% = head, used later for RQ1)
    q80 = art["n_listeners"].quantile(0.80)
    art["popularity_bucket"] = np.where(art["n_listeners"] >= q80, "head", "tail")

    # 4. tags
    at = clean_tags(uta, tags, keep)
    topk = top_k_tags(at)
    art = art.merge(topk[["artistID"]].assign(has_tags=True), on="artistID", how="left")
    art["has_tags"] = art["has_tags"].fillna(False).astype(bool)

    # 5. save
    inter.to_csv(PROCESSED_DIR / "interactions.csv", index=False)
    art.to_csv(PROCESSED_DIR / "artists_filtered.csv", index=False)
    at.to_csv(PROCESSED_DIR / "artist_tags.csv", index=False)
    topk.to_csv(PROCESSED_DIR / "artist_tags_topk.csv", index=False)

    print(f"Users: {inter['userID'].nunique():,}  Interactions: {len(inter):,}")
    print(f"Artists with >=1 clean tag: {art['has_tags'].mean():.1%}")
    print(f"Distinct clean tags: {at['tag'].nunique():,} (raw used: {uta['tagID'].nunique():,})")
    print("Saved to", PROCESSED_DIR)


if __name__ == "__main__":
    main()
