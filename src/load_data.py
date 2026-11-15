"""Load the raw HetRec 2011 Last.fm files into pandas DataFrames."""
import pandas as pd

from config import RAW_DIR


def _read(name, encoding="utf-8"):
    return pd.read_csv(RAW_DIR / name, sep="\t", encoding=encoding)


def load_artists():
    return _read("artists.dat").rename(columns={"id": "artistID"})


def load_tags():
    # tags.dat is not valid UTF-8, it is Latin-1
    return _read("tags.dat", encoding="latin-1")


def load_user_artists():
    return _read("user_artists.dat")


def load_user_tagged_artists():
    return _read("user_taggedartists.dat")


def load_user_friends():
    return _read("user_friends.dat")


def load_all():
    return {
        "artists": load_artists(),
        "tags": load_tags(),
        "user_artists": load_user_artists(),
        "user_taggedartists": load_user_tagged_artists(),
        "user_friends": load_user_friends(),
    }
