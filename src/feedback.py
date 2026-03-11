"""Lightweight feedback capture - persists to local JSONL file."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict

from src.config import FEEDBACK_PATH


def save_feedback(
    feedback_type: str,
    entity_id: str = "",
    narrative_id: str = "",
    post_id: str = "",
    details: dict = None,
) -> None:
    """Append a feedback entry to feedback.jsonl.

    feedback_type: "entity_correction" | "risk_rating" | "narrative_feedback"
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feedback_type": feedback_type,
        "entity_id": entity_id,
        "narrative_id": narrative_id,
        "post_id": post_id,
        "details": details or {},
    }
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_feedback() -> List[Dict]:
    """Load all feedback entries."""
    if not FEEDBACK_PATH.exists():
        return []
    entries = []
    with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
