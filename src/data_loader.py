"""Load and prepare the raw data files."""

import json
import pandas as pd
from pathlib import Path
from typing import Tuple

from src.config import POSTS_PATH, AUTHORS_PATH, ENTITIES_PATH


def load_posts(path: Path = POSTS_PATH) -> pd.DataFrame:
    """Load posts.jsonl into a DataFrame."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    df = pd.DataFrame(records)
    df["created_at"] = pd.to_datetime(df["created_at"])
    # Fill missing engagement fields with 0
    for col in ["likes", "shares", "comments", "views"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def load_authors(path: Path = AUTHORS_PATH) -> pd.DataFrame:
    """Load authors.csv into a DataFrame."""
    df = pd.read_csv(path, dtype={"author_id": str})
    df["followers"] = pd.to_numeric(df["followers"], errors="coerce").fillna(0).astype(int)
    return df


def load_entities(path: Path = ENTITIES_PATH) -> pd.DataFrame:
    """Load entities_seed.csv into a DataFrame."""
    df = pd.read_csv(path, dtype=str).fillna("")
    return df


def build_entity_lookup(entities_df: pd.DataFrame) -> dict:
    """Build a lookup dict: entity_id -> {canonical_name, entity_type, aliases: list}."""
    lookup = {}
    for _, row in entities_df.iterrows():
        aliases = []
        if row["aliases"]:
            aliases = [a.strip() for a in row["aliases"].split("|") if a.strip()]
        # Always include the canonical name as an alias
        all_names = [row["canonical_name"]] + aliases
        lookup[row["entity_id"]] = {
            "canonical_name": row["canonical_name"],
            "entity_type": row.get("entity_type", ""),
            "aliases": all_names,
        }
    return lookup


def load_all() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load all data and return (posts, authors, entities_df, entity_lookup)."""
    posts = load_posts()
    authors = load_authors()
    entities = load_entities()
    lookup = build_entity_lookup(entities)

    # Merge author info into posts
    posts = posts.merge(
        authors[["author_id", "handle", "followers"]].astype({"author_id": str}),
        on="author_id",
        how="left",
    )
    posts["followers"] = posts["followers"].fillna(0).astype(int)

    return posts, authors, entities, lookup
