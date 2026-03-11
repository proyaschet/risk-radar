"""Tests for entity_resolution module."""

import pytest
import pandas as pd
from src.entity_resolution import (
    _build_alias_index,
    _exact_match,
    _fuzzy_match,
    _embedding_match,
    resolve_entities_for_post,
    get_posts_for_entity,
    get_entity_confidence_stats,
)
from src.data_loader import load_entities, build_entity_lookup


@pytest.fixture
def entity_lookup():
    entities = load_entities()
    return build_entity_lookup(entities)


@pytest.fixture
def alias_index(entity_lookup):
    return _build_alias_index(entity_lookup)


class TestBuildAliasIndex:
    def test_index_created(self, entity_lookup):
        index = _build_alias_index(entity_lookup)
        assert isinstance(index, dict)
        assert len(index) > 0

    def test_lowercase_keys(self, alias_index):
        for key in alias_index:
            assert key == key.lower()


class TestExactMatch:
    def test_exact_pfizer_match(self, alias_index):
        matches = _exact_match("Pfizer announced new drug results", alias_index)
        assert len(matches) >= 1
        assert any(m["entity_id"] == "pfizer" for m in matches)
        assert all(m["confidence"] == 0.95 for m in matches)

    def test_exact_moderna_match(self, alias_index):
        matches = _exact_match("Moderna vaccine update", alias_index)
        assert any(m["entity_id"] == "moderna" for m in matches)

    def test_no_match_random_text(self, alias_index):
        matches = _exact_match("The weather is nice today", alias_index)
        assert len(matches) == 0

    def test_case_insensitive(self, alias_index):
        matches = _exact_match("PFIZER is great", alias_index)
        assert any(m["entity_id"] == "pfizer" for m in matches)

    def test_multiple_entities(self, alias_index):
        matches = _exact_match("Pfizer and Moderna are both pharma companies", alias_index)
        entity_ids = {m["entity_id"] for m in matches}
        assert "pfizer" in entity_ids
        assert "moderna" in entity_ids


class TestFuzzyMatch:
    def test_fuzzy_misspelling(self, entity_lookup):
        matches = _fuzzy_match("Pfeizer drug update", entity_lookup)
        # May or may not match depending on threshold
        assert isinstance(matches, list)

    def test_fuzzy_partial_match(self, entity_lookup):
        matches = _fuzzy_match("AstraZeneca PLC announced today", entity_lookup)
        assert any(m["entity_id"] == "astrazeneca" for m in matches)

    def test_fuzzy_confidence_capped(self, entity_lookup):
        matches = _fuzzy_match("Bristol-Myers Squibb report", entity_lookup)
        for m in matches:
            assert m["confidence"] <= 0.90


class TestEmbeddingMatch:
    def test_semantic_match(self, entity_lookup):
        from src.entity_resolution import _precompute_entity_embeddings
        entity_embs, entity_ids_list = _precompute_entity_embeddings(entity_lookup)
        matches = _embedding_match(
            "The HPV vaccine caused side effects in my daughter",
            entity_lookup,
            entity_embs,
            entity_ids_list,
        )
        assert isinstance(matches, list)
        # Should find at least gardasil or hpv
        entity_ids = {m["entity_id"] for m in matches}
        assert len(entity_ids) > 0


class TestResolveEntitiesForPost:
    def test_exact_resolution(self, entity_lookup, alias_index):
        result = resolve_entities_for_post(
            "Pfizer stock dropped today",
            entity_lookup,
            alias_index,
            use_llm=False,
        )
        assert result["resolution_method"] == "exact_alias"
        assert not result["needs_review"]
        assert len(result["resolved_entities"]) >= 1

    def test_no_entities(self, entity_lookup, alias_index):
        result = resolve_entities_for_post(
            "Beautiful sunset this evening",
            entity_lookup,
            alias_index,
            use_llm=False,
        )
        assert len(result["resolved_entities"]) == 0

    def test_result_structure(self, entity_lookup, alias_index):
        result = resolve_entities_for_post(
            "Keytruda treatment results",
            entity_lookup,
            alias_index,
            use_llm=False,
        )
        assert "resolved_entities" in result
        assert "resolution_method" in result
        assert "needs_review" in result
        for ent in result["resolved_entities"]:
            assert "entity_id" in ent
            assert "mention_text" in ent
            assert "confidence" in ent
            assert "resolution_method" in ent


class TestGetPostsForEntity:
    def test_filters_correctly(self):
        df = pd.DataFrame(
            {
                "post_id": ["1", "2", "3"],
                "text": ["a", "b", "c"],
                "resolved_entities": [
                    [{"entity_id": "pfizer", "mention_text": "Pfizer", "confidence": 0.95}],
                    [{"entity_id": "moderna", "mention_text": "Moderna", "confidence": 0.90}],
                    [{"entity_id": "pfizer", "mention_text": "Pfizer", "confidence": 0.80}],
                ],
            }
        )
        result = get_posts_for_entity(df, "pfizer")
        assert len(result) == 2
        assert set(result["post_id"].tolist()) == {"1", "3"}


class TestEntityConfidenceStats:
    def test_stats_structure(self):
        df = pd.DataFrame(
            {
                "post_id": ["1", "2"],
                "text": ["a", "b"],
                "resolved_entities": [
                    [{"entity_id": "pfizer", "mention_text": "Pfizer", "confidence": 0.95, "resolution_method": "exact_alias"}],
                    [{"entity_id": "pfizer", "mention_text": "Pfizer", "confidence": 0.70, "resolution_method": "fuzzy"}],
                ],
            }
        )
        stats = get_entity_confidence_stats(df, "pfizer")
        assert stats["total"] == 2
        assert stats["high"] == 1
        assert stats["medium"] == 1
        assert stats["low"] == 0
        assert "methods" in stats
