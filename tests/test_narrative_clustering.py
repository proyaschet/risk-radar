"""Tests for narrative_clustering module."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.narrative_clustering import (
    _cluster_posts,
    _extractive_title,
    _extractive_summary,
    _classify_taxonomy_local,
    cluster_narratives,
)


@pytest.fixture
def sample_entity_posts():
    """Create posts that should form at least 2 distinct clusters."""
    now = datetime.now()
    # Cluster 1: Safety/side effects posts
    safety_posts = [
        "Severe side effects reported after taking this medication",
        "Patients experiencing adverse reactions to the drug",
        "FDA investigating reports of harmful side effects",
        "Multiple complaints about dangerous adverse effects from treatment",
        "Doctor reports injury and side effects in patients",
    ]
    # Cluster 2: Financial/stock posts
    financial_posts = [
        "Stock price dropped significantly after earnings report",
        "Revenue missed estimates by wide margin this quarter",
        "Investors selling shares after poor financial results",
        "Market analysts downgrade stock rating after losses",
        "Shareholders concerned about declining profits and revenue",
    ]
    texts = safety_posts + financial_posts

    return pd.DataFrame(
        {
            "post_id": [f"p{i}" for i in range(len(texts))],
            "text": texts,
            "created_at": [now - timedelta(hours=i) for i in range(len(texts))],
            "likes": [10 + i for i in range(len(texts))],
            "shares": [5 + i for i in range(len(texts))],
            "comments": [2 + i for i in range(len(texts))],
            "views": [100 + i * 50 for i in range(len(texts))],
            "platform": ["twitter"] * 5 + ["facebook"] * 5,
            "author_id": [f"a{i}" for i in range(len(texts))],
            "followers": [100 * (i + 1) for i in range(len(texts))],
        }
    )


class TestClusterPosts:
    def test_returns_labels(self):
        embeddings = np.random.rand(10, 384)
        labels = _cluster_posts(embeddings)
        assert len(labels) == 10

    def test_small_input(self):
        embeddings = np.random.rand(2, 384)
        labels = _cluster_posts(embeddings, min_size=3)
        assert len(labels) == 2

    def test_single_post(self):
        embeddings = np.random.rand(1, 384)
        labels = _cluster_posts(embeddings)
        assert len(labels) == 1


class TestExtractiveMethods:
    def test_title_generation(self, sample_entity_posts):
        title = _extractive_title(sample_entity_posts)
        assert isinstance(title, str)
        assert len(title) > 0
        assert len(title) <= 123  # 120 + "..."

    def test_summary_generation(self, sample_entity_posts):
        summary = _extractive_summary(sample_entity_posts)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_empty_posts(self):
        empty = pd.DataFrame({"text": [], "likes": [], "shares": [], "comments": []})
        assert _extractive_title(empty) == "Unknown Narrative"
        assert _extractive_summary(empty) == "No posts available."


class TestClassifyTaxonomyLocal:
    def test_customer_harm(self):
        df = pd.DataFrame({"text": ["Patient suffered severe side effects and injury"]})
        labels = _classify_taxonomy_local(df)
        assert "customer_harm" in labels

    def test_regulatory(self):
        df = pd.DataFrame({"text": ["FDA investigation into compliance violation and lawsuit"]})
        labels = _classify_taxonomy_local(df)
        assert "regulatory_compliance" in labels

    def test_misinfo(self):
        df = pd.DataFrame({"text": ["This is a conspiracy by big pharma to cover-up the truth"]})
        labels = _classify_taxonomy_local(df)
        assert "misinfo_manipulation" in labels

    def test_multiple_labels(self):
        df = pd.DataFrame({"text": [
            "Patient death after fraud investigation by FDA into data breach and leaked documents"
        ]})
        labels = _classify_taxonomy_local(df)
        assert len(labels) >= 2

    def test_default_label(self):
        df = pd.DataFrame({"text": ["Beautiful sunny day at the park"]})
        labels = _classify_taxonomy_local(df)
        assert len(labels) >= 1  # Should get default


class TestClusterNarratives:
    def test_returns_list(self, sample_entity_posts):
        narratives = cluster_narratives(
            sample_entity_posts, "pfizer", "Pfizer", use_llm=False
        )
        assert isinstance(narratives, list)

    def test_narrative_structure(self, sample_entity_posts):
        narratives = cluster_narratives(
            sample_entity_posts, "pfizer", "Pfizer", use_llm=False
        )
        for n in narratives:
            assert "narrative_id" in n
            assert "entity_id" in n
            assert "title" in n
            assert "summary" in n
            assert "post_ids" in n
            assert "post_count" in n
            assert "taxonomy_labels" in n
            assert "member_posts" in n
            assert n["entity_id"] == "pfizer"

    def test_empty_posts(self):
        empty = pd.DataFrame(
            columns=["post_id", "text", "created_at", "likes", "shares", "comments", "views", "platform", "author_id", "followers"]
        )
        narratives = cluster_narratives(empty, "pfizer", "Pfizer", use_llm=False)
        assert narratives == []

    def test_posts_assigned_to_narratives(self, sample_entity_posts):
        narratives = cluster_narratives(
            sample_entity_posts, "pfizer", "Pfizer", use_llm=False
        )
        all_post_ids = set()
        for n in narratives:
            all_post_ids.update(n["post_ids"])
        expected = set(sample_entity_posts["post_id"].tolist())
        # Most posts should be in some narrative; small clusters may be dropped
        assert len(all_post_ids) >= len(expected) * 0.7
