"""Pre-compute entity resolution and cache results to disk for fast app startup."""

import json
import pickle
from pathlib import Path

from src.data_loader import load_all
from src.entity_resolution import resolve_all_posts
from src.config import OUTPUT_DIR, USE_LLM

CACHE_PATH = OUTPUT_DIR / "resolved_posts.pkl"


def precompute():
    print("Loading data...")
    posts, authors, entities, lookup = load_all()
    print(f"  Posts: {len(posts)}, Authors: {len(authors)}, Entities: {len(lookup)}")

    print("Resolving entities (this takes ~30s first time)...")
    posts = resolve_all_posts(posts, lookup, use_llm=USE_LLM)

    # Count matches
    matched = posts["resolved_entities"].apply(lambda x: len(x) > 0).sum()
    print(f"  Matched: {matched}/{len(posts)} posts")

    print(f"Saving cache to {CACHE_PATH}...")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"posts": posts, "authors": authors, "entities": entities, "lookup": lookup}, f)

    print("Done! App will now start instantly.")


if __name__ == "__main__":
    precompute()
