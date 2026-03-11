"""Tests for feedback module."""

import json
import pytest
from pathlib import Path
from src.feedback import save_feedback, load_feedback
from src.config import FEEDBACK_PATH


@pytest.fixture(autouse=True)
def clean_feedback():
    """Remove feedback file before and after each test."""
    test_path = FEEDBACK_PATH.parent / "test_feedback.jsonl"
    yield test_path
    if test_path.exists():
        test_path.unlink()


class TestFeedback:
    def test_save_and_load(self, clean_feedback, monkeypatch):
        monkeypatch.setattr("src.feedback.FEEDBACK_PATH", clean_feedback)
        monkeypatch.setattr("src.config.FEEDBACK_PATH", clean_feedback)

        save_feedback(
            feedback_type="entity_correction",
            entity_id="pfizer",
            post_id="123",
            details={"issue": "Incorrect entity match", "correct_entity": "moderna"},
        )
        entries = load_feedback()
        assert len(entries) == 1
        assert entries[0]["feedback_type"] == "entity_correction"
        assert entries[0]["entity_id"] == "pfizer"
        assert entries[0]["details"]["correct_entity"] == "moderna"

    def test_multiple_entries(self, clean_feedback, monkeypatch):
        monkeypatch.setattr("src.feedback.FEEDBACK_PATH", clean_feedback)
        monkeypatch.setattr("src.config.FEEDBACK_PATH", clean_feedback)

        save_feedback(feedback_type="risk_rating", entity_id="pfizer", details={"user_score": 80})
        save_feedback(feedback_type="risk_rating", entity_id="moderna", details={"user_score": 30})

        entries = load_feedback()
        assert len(entries) == 2

    def test_load_empty(self, clean_feedback, monkeypatch):
        monkeypatch.setattr("src.feedback.FEEDBACK_PATH", clean_feedback)
        entries = load_feedback()
        assert entries == []

    def test_entry_has_timestamp(self, clean_feedback, monkeypatch):
        monkeypatch.setattr("src.feedback.FEEDBACK_PATH", clean_feedback)
        monkeypatch.setattr("src.config.FEEDBACK_PATH", clean_feedback)

        save_feedback(feedback_type="test")
        entries = load_feedback()
        assert "timestamp" in entries[0]
