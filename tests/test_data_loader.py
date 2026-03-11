"""Tests for data_loader module."""

import pandas as pd
import pytest
from pathlib import Path
from src.data_loader import (
    load_posts,
    load_authors,
    load_entities,
    build_entity_lookup,
    load_all,
)
from src.config import POSTS_PATH, AUTHORS_PATH, ENTITIES_PATH


class TestLoadPosts:
    def test_loads_dataframe(self):
        df = load_posts()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self):
        df = load_posts()
        required = ["post_id", "created_at", "platform", "author_id", "text"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_datetime_parsed(self):
        df = load_posts()
        assert pd.api.types.is_datetime64_any_dtype(df["created_at"])

    def test_engagement_numeric(self):
        df = load_posts()
        for col in ["likes", "shares", "comments", "views"]:
            if col in df.columns:
                assert pd.api.types.is_integer_dtype(df[col])


class TestLoadAuthors:
    def test_loads_dataframe(self):
        df = load_authors()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self):
        df = load_authors()
        assert "author_id" in df.columns
        assert "followers" in df.columns

    def test_followers_numeric(self):
        df = load_authors()
        assert pd.api.types.is_integer_dtype(df["followers"])


class TestLoadEntities:
    def test_loads_dataframe(self):
        df = load_entities()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 19  # 19 entities in seed file

    def test_required_columns(self):
        df = load_entities()
        for col in ["entity_id", "canonical_name", "entity_type"]:
            assert col in df.columns


class TestBuildEntityLookup:
    def test_lookup_structure(self):
        entities = load_entities()
        lookup = build_entity_lookup(entities)
        assert isinstance(lookup, dict)
        assert len(lookup) == 19

    def test_lookup_values(self):
        entities = load_entities()
        lookup = build_entity_lookup(entities)
        for eid, info in lookup.items():
            assert "canonical_name" in info
            assert "entity_type" in info
            assert "aliases" in info
            assert isinstance(info["aliases"], list)
            assert len(info["aliases"]) >= 1  # At least canonical name

    def test_pfizer_aliases(self):
        entities = load_entities()
        lookup = build_entity_lookup(entities)
        assert "pfizer" in lookup
        assert "Pfizer" in lookup["pfizer"]["aliases"]


class TestLoadAll:
    def test_returns_tuple(self):
        posts, authors, entities, lookup = load_all()
        assert isinstance(posts, pd.DataFrame)
        assert isinstance(authors, pd.DataFrame)
        assert isinstance(entities, pd.DataFrame)
        assert isinstance(lookup, dict)

    def test_author_merge(self):
        posts, _, _, _ = load_all()
        assert "followers" in posts.columns
        assert "handle" in posts.columns
