"""Tests for risk_scoring module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.risk_scoring import (
    _score_volume_velocity,
    _score_engagement,
    _score_author_influence,
    _score_language_risk,
    _score_virality,
    compute_risk_score,
    score_all_narratives,
    get_evidence_posts,
)


@pytest.fixture
def sample_posts():
    """Create a sample posts DataFrame for testing."""
    now = datetime.now()
    return pd.DataFrame(
        {
            "post_id": [f"p{i}" for i in range(10)],
            "text": [
                "Pfizer lawsuit fraud scandal dangerous",
                "Side effects reported by many patients",
                "Concerned about the safety of this drug",
                "Investigation into misconduct allegations",
                "Death reported after taking medication",
                "Normal post about pharma industry",
                "Another safety concern raised",
                "FDA looking into the issue",
                "Controversy over clinical trial results",
                "Patients complaining about adverse effects",
            ],
            "created_at": [now - timedelta(hours=i) for i in range(10)],
            "likes": [100, 50, 30, 20, 200, 5, 15, 40, 60, 25],
            "shares": [50, 20, 10, 5, 100, 1, 5, 15, 30, 10],
            "comments": [30, 15, 8, 3, 80, 2, 5, 10, 20, 8],
            "views": [1000, 500, 200, 100, 5000, 50, 100, 300, 800, 200],
            "platform": ["twitter"] * 5 + ["facebook"] * 5,
            "author_id": [f"a{i}" for i in range(10)],
            "followers": [50000, 1000, 500, 200, 100000, 50, 100, 5000, 20000, 300],
        }
    )


@pytest.fixture
def sample_narrative(sample_posts):
    return {
        "narrative_id": "test_narrative_1",
        "entity_id": "pfizer",
        "title": "Test Narrative",
        "summary": "Test summary",
        "post_ids": sample_posts["post_id"].tolist(),
        "post_count": len(sample_posts),
        "taxonomy_labels": ["customer_harm"],
        "key_themes": ["safety", "side effects"],
        "member_posts": sample_posts,
    }


class TestVolumeVelocity:
    def test_empty_posts(self):
        score, explanation = _score_volume_velocity(pd.DataFrame())
        assert score == 0.0

    def test_positive_with_posts(self, sample_posts):
        score, explanation = _score_volume_velocity(sample_posts)
        assert 0 < score <= 100
        assert "10 posts" in explanation

    def test_more_posts_higher_score(self):
        small = pd.DataFrame({"created_at": pd.date_range("2025-01-01", periods=3, freq="h")})
        large = pd.DataFrame({"created_at": pd.date_range("2025-01-01", periods=30, freq="h")})
        score_small, _ = _score_volume_velocity(small)
        score_large, _ = _score_volume_velocity(large)
        assert score_large > score_small


class TestEngagement:
    def test_empty(self):
        score, _ = _score_engagement(pd.DataFrame())
        assert score == 0.0

    def test_positive(self, sample_posts):
        score, explanation = _score_engagement(sample_posts)
        assert 0 < score <= 100
        assert "likes" in explanation

    def test_high_shares_boost(self):
        df = pd.DataFrame({"likes": [10], "shares": [100], "comments": [5], "views": [1000]})
        score, _ = _score_engagement(df)
        assert score > 0


class TestAuthorInfluence:
    def test_empty(self):
        score, _ = _score_author_influence(pd.DataFrame())
        assert score == 0.0

    def test_high_influence(self, sample_posts):
        score, explanation = _score_author_influence(sample_posts)
        assert score > 0
        assert "high-influence" in explanation  # We have authors with >10k followers

    def test_no_followers(self):
        df = pd.DataFrame({"author_id": ["a1"], "followers": [0]})
        score, _ = _score_author_influence(df)
        assert score == 0.0


class TestLanguageRisk:
    def test_empty(self):
        score, _ = _score_language_risk(pd.DataFrame())
        assert score == 0.0

    def test_high_risk_language(self):
        df = pd.DataFrame({"text": ["fraud scandal death lawsuit criminal investigation"]})
        score, explanation = _score_language_risk(df)
        assert score > 50
        assert "high-risk terms" in explanation

    def test_benign_text(self):
        df = pd.DataFrame({"text": ["The weather is beautiful today and flowers are blooming"]})
        score, explanation = _score_language_risk(df)
        assert score < 20

    def test_mixed_risk(self, sample_posts):
        score, _ = _score_language_risk(sample_posts)
        assert 0 < score <= 100


class TestVirality:
    def test_empty(self):
        score, _ = _score_virality(pd.DataFrame())
        assert score == 0.0

    def test_cross_platform(self, sample_posts):
        score, explanation = _score_virality(sample_posts)
        assert score > 0
        assert "2 platform(s)" in explanation

    def test_single_platform(self):
        df = pd.DataFrame({
            "likes": [10], "shares": [5], "platform": ["twitter"],
        })
        score, _ = _score_virality(df)
        assert score >= 0


class TestComputeRiskScore:
    def test_score_structure(self, sample_narrative):
        result = compute_risk_score(sample_narrative)
        assert "risk_score" in result
        assert "risk_confidence" in result
        assert "risk_level" in result
        assert "risk_drivers" in result
        assert 0 <= result["risk_score"] <= 100

    def test_drivers_present(self, sample_narrative):
        result = compute_risk_score(sample_narrative)
        assert len(result["risk_drivers"]) == 5
        driver_names = {d["driver"] for d in result["risk_drivers"]}
        assert "Volume & Velocity" in driver_names
        assert "Language Risk Signals" in driver_names

    def test_driver_structure(self, sample_narrative):
        result = compute_risk_score(sample_narrative)
        for driver in result["risk_drivers"]:
            assert "driver" in driver
            assert "score" in driver
            assert "weight" in driver
            assert "weighted_contribution" in driver
            assert "explanation" in driver

    def test_high_risk_posts_score_high(self, sample_narrative):
        result = compute_risk_score(sample_narrative)
        # Our sample has lots of risky language + high engagement
        assert result["risk_score"] > 30


class TestScoreAllNarratives:
    def test_sorted_descending(self, sample_posts):
        narratives = [
            {
                "narrative_id": "n1",
                "entity_id": "pfizer",
                "title": "Benign",
                "summary": "ok",
                "post_ids": ["p0"],
                "post_count": 1,
                "taxonomy_labels": [],
                "key_themes": [],
                "member_posts": sample_posts.head(1),
            },
            {
                "narrative_id": "n2",
                "entity_id": "pfizer",
                "title": "Risky",
                "summary": "bad",
                "post_ids": sample_posts["post_id"].tolist(),
                "post_count": len(sample_posts),
                "taxonomy_labels": ["customer_harm"],
                "key_themes": ["safety"],
                "member_posts": sample_posts,
            },
        ]
        scored = score_all_narratives(narratives)
        assert scored[0]["risk_score"] >= scored[1]["risk_score"]


class TestGetEvidencePosts:
    def test_returns_top_n(self, sample_narrative):
        evidence = get_evidence_posts(sample_narrative, top_n=3)
        assert len(evidence) == 3

    def test_sorted_by_engagement(self, sample_narrative):
        evidence = get_evidence_posts(sample_narrative, top_n=5)
        engagements = evidence["total_engagement"].tolist()
        assert engagements == sorted(engagements, reverse=True)
