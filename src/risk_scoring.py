"""Narrative risk scoring (0-100) with explainability.

Score = weighted sum of normalised signal dimensions:
  - Volume & velocity (20%): post count + posting rate acceleration
  - Engagement amplification (20%): total likes/shares/comments/views normalised
  - Author influence (15%): follower-weighted reach, presence of high-influence authors
  - Language risk signals (30%): taxonomy-aligned keyword density, sentiment proxies
  - Virality indicators (15%): share-to-like ratio, cross-platform spread

Each driver returns a sub-score (0-100) and a human-readable explanation.
Final score = weighted combination, with confidence interval.

Design rationale:
- Language risk has highest weight because content determines reputational harm
- Engagement and volume are necessary but not sufficient (high volume + benign = low risk)
- All weights are explicit and auditable; no hidden "magic numbers"
"""

import logging
from typing import Dict, List, Tuple
from datetime import timedelta

import numpy as np
import pandas as pd

from src.config import RISK_WEIGHTS, RISK_TAXONOMY

logger = logging.getLogger(__name__)

# Language risk keywords grouped by severity
RISK_KEYWORDS = {
    "high": [
        "fraud", "lawsuit", "death", "died", "kill", "cancer", "criminal",
        "investigation", "scandal", "corrupt", "cover-up", "banned",
        "recalled", "dangerous", "toxic", "poisoned", "genocide",
    ],
    "medium": [
        "side effect", "adverse", "complaint", "unsafe", "risk", "warning",
        "allegation", "controversy", "concern", "injury", "harm",
        "misleading", "conspiracy", "hoax", "fake", "shill", "propaganda",
    ],
    "low": [
        "question", "doubt", "worried", "concerned", "issue", "problem",
        "disappointed", "alternative", "natural", "organic",
    ],
}


def _score_volume_velocity(posts: pd.DataFrame) -> Tuple[float, str]:
    """Score based on post count and posting rate acceleration."""
    n_posts = len(posts)
    if n_posts == 0:
        return 0.0, "No posts"

    # Volume component (log-scaled, max ~50 posts for full score)
    volume_score = min(np.log1p(n_posts) / np.log1p(50) * 100, 100)

    # Velocity: are posts accelerating recently?
    velocity_detail = ""
    if "created_at" in posts.columns and n_posts >= 3:
        posts_sorted = posts.sort_values("created_at")
        dates = posts_sorted["created_at"]
        total_span = (dates.max() - dates.min()).total_seconds() / 3600  # hours

        if total_span > 0:
            # Compare first half vs second half posting rate
            mid = dates.median()
            first_half = len(dates[dates <= mid])
            second_half = len(dates[dates > mid])

            if first_half > 0:
                acceleration = second_half / max(first_half, 1)
                if acceleration > 1.5:
                    volume_score = min(volume_score * 1.3, 100)
                    velocity_detail = f" (accelerating: {acceleration:.1f}x recent vs earlier)"
                elif acceleration < 0.5:
                    volume_score *= 0.8
                    velocity_detail = " (decelerating)"

    explanation = f"{n_posts} posts{velocity_detail}"
    return round(volume_score, 1), explanation


def _score_engagement(posts: pd.DataFrame) -> Tuple[float, str]:
    """Score based on engagement metrics."""
    if posts.empty:
        return 0.0, "No engagement data"

    total_likes = posts["likes"].sum()
    total_shares = posts["shares"].sum()
    total_comments = posts["comments"].sum()
    total_views = posts["views"].sum()
    total_engagement = total_likes + total_shares * 2 + total_comments * 1.5

    # Log-scaled normalisation
    score = min(np.log1p(total_engagement) / np.log1p(5000) * 100, 100)

    # Shares are especially important for amplification
    if total_shares > 10:
        score = min(score * 1.2, 100)

    parts = []
    if total_likes > 0:
        parts.append(f"{total_likes} likes")
    if total_shares > 0:
        parts.append(f"{total_shares} shares")
    if total_comments > 0:
        parts.append(f"{total_comments} comments")
    if total_views > 0:
        parts.append(f"{total_views:,} views")

    explanation = ", ".join(parts) if parts else "Minimal engagement"
    return round(score, 1), explanation


def _score_author_influence(posts: pd.DataFrame) -> Tuple[float, str]:
    """Score based on author follower counts and reach."""
    if posts.empty or "followers" not in posts.columns:
        return 0.0, "No author data"

    followers = posts["followers"].fillna(0)
    max_followers = followers.max()
    total_reach = followers.sum()
    n_authors = posts["author_id"].nunique()

    # High-influence authors (>10k followers)
    high_influence = len(followers[followers > 10000])

    # Log-scaled score
    score = min(np.log1p(total_reach) / np.log1p(500000) * 100, 100)

    if high_influence > 0:
        score = min(score * 1.3, 100)

    parts = [f"{n_authors} unique authors"]
    if max_followers > 0:
        parts.append(f"max {max_followers:,} followers")
    if high_influence > 0:
        parts.append(f"{high_influence} high-influence (>10k)")
    parts.append(f"total reach {total_reach:,}")

    return round(score, 1), ", ".join(parts)


def _score_language_risk(posts: pd.DataFrame) -> Tuple[float, str]:
    """Score based on risk-aligned language signals."""
    if posts.empty:
        return 0.0, "No text to analyze"

    all_text = " ".join(posts["text"].fillna("").str.lower())
    n_posts = len(posts)

    high_hits = []
    medium_hits = []
    low_hits = []

    for kw in RISK_KEYWORDS["high"]:
        count = all_text.count(kw)
        if count > 0:
            high_hits.append((kw, count))

    for kw in RISK_KEYWORDS["medium"]:
        count = all_text.count(kw)
        if count > 0:
            medium_hits.append((kw, count))

    for kw in RISK_KEYWORDS["low"]:
        count = all_text.count(kw)
        if count > 0:
            low_hits.append((kw, count))

    # Weighted keyword density
    high_count = sum(c for _, c in high_hits)
    medium_count = sum(c for _, c in medium_hits)
    low_count = sum(c for _, c in low_hits)

    weighted = high_count * 3 + medium_count * 1.5 + low_count * 0.5
    # Normalise per post, cap at reasonable max
    density = weighted / max(n_posts, 1)
    score = min(density / 5.0 * 100, 100)  # 5+ weighted keywords per post = max

    parts = []
    if high_hits:
        top_high = sorted(high_hits, key=lambda x: -x[1])[:3]
        parts.append(f"high-risk terms: {', '.join(f'{kw}({c})' for kw, c in top_high)}")
    if medium_hits:
        top_med = sorted(medium_hits, key=lambda x: -x[1])[:3]
        parts.append(f"medium-risk terms: {', '.join(f'{kw}({c})' for kw, c in top_med)}")
    if not parts:
        parts.append("No significant risk language detected")

    return round(score, 1), "; ".join(parts)


def _score_virality(posts: pd.DataFrame) -> Tuple[float, str]:
    """Score based on virality indicators: share ratio, cross-platform spread."""
    if posts.empty:
        return 0.0, "No virality data"

    total_shares = posts["shares"].sum()
    total_likes = posts["likes"].sum()
    n_platforms = posts["platform"].nunique() if "platform" in posts.columns else 1

    # Share-to-like ratio (high = content being actively amplified)
    share_ratio = total_shares / max(total_likes, 1)
    ratio_score = min(share_ratio * 50, 60)  # 2:1 share:like = 100

    # Cross-platform spread bonus
    platform_bonus = min((n_platforms - 1) * 20, 40)

    score = min(ratio_score + platform_bonus, 100)

    parts = []
    if total_shares > 0 and total_likes > 0:
        parts.append(f"share:like ratio {share_ratio:.2f}")
    parts.append(f"{n_platforms} platform(s)")

    return round(score, 1), ", ".join(parts)


def compute_risk_score(narrative: Dict) -> Dict:
    """Compute risk score for a narrative. Returns enriched narrative dict.

    Adds:
        risk_score: float (0-100)
        risk_confidence: str (low/medium/high)
        risk_drivers: list of {driver, score, weight, explanation}
        risk_level: str (critical/high/medium/low)
    """
    posts = narrative["member_posts"]

    # Compute each driver
    vol_score, vol_explain = _score_volume_velocity(posts)
    eng_score, eng_explain = _score_engagement(posts)
    auth_score, auth_explain = _score_author_influence(posts)
    lang_score, lang_explain = _score_language_risk(posts)
    viral_score, viral_explain = _score_virality(posts)

    drivers = [
        {
            "driver": "Volume & Velocity",
            "score": vol_score,
            "weight": RISK_WEIGHTS["volume_velocity"],
            "weighted_contribution": round(vol_score * RISK_WEIGHTS["volume_velocity"], 1),
            "explanation": vol_explain,
        },
        {
            "driver": "Engagement & Amplification",
            "score": eng_score,
            "weight": RISK_WEIGHTS["engagement"],
            "weighted_contribution": round(eng_score * RISK_WEIGHTS["engagement"], 1),
            "explanation": eng_explain,
        },
        {
            "driver": "Author Influence",
            "score": auth_score,
            "weight": RISK_WEIGHTS["author_influence"],
            "weighted_contribution": round(auth_score * RISK_WEIGHTS["author_influence"], 1),
            "explanation": auth_explain,
        },
        {
            "driver": "Language Risk Signals",
            "score": lang_score,
            "weight": RISK_WEIGHTS["language_risk"],
            "weighted_contribution": round(lang_score * RISK_WEIGHTS["language_risk"], 1),
            "explanation": lang_explain,
        },
        {
            "driver": "Virality Indicators",
            "score": viral_score,
            "weight": RISK_WEIGHTS["virality"],
            "weighted_contribution": round(viral_score * RISK_WEIGHTS["virality"], 1),
            "explanation": viral_explain,
        },
    ]

    # Weighted sum
    risk_score = sum(d["weighted_contribution"] for d in drivers)
    risk_score = round(min(max(risk_score, 0), 100), 1)

    # Confidence based on sample size and signal consistency
    n_posts = len(posts)
    score_variance = np.var([d["score"] for d in drivers])
    if n_posts >= 10 and score_variance < 500:
        confidence = "high"
    elif n_posts >= 5:
        confidence = "medium"
    else:
        confidence = "low"

    # Risk level
    if risk_score >= 70:
        risk_level = "critical"
    elif risk_score >= 50:
        risk_level = "high"
    elif risk_score >= 30:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Sort drivers by weighted contribution (highest first)
    drivers.sort(key=lambda d: -d["weighted_contribution"])

    narrative["risk_score"] = risk_score
    narrative["risk_confidence"] = confidence
    narrative["risk_level"] = risk_level
    narrative["risk_drivers"] = drivers

    return narrative


def score_all_narratives(narratives: List[Dict]) -> List[Dict]:
    """Score all narratives and return sorted by risk score (descending)."""
    scored = [compute_risk_score(n) for n in narratives]
    scored.sort(key=lambda n: -n["risk_score"])
    return scored


def get_evidence_posts(narrative: Dict, top_n: int = 5) -> pd.DataFrame:
    """Get top evidence posts for a narrative, ranked by engagement."""
    posts = narrative["member_posts"]
    if posts.empty:
        return posts

    posts = posts.copy()
    posts["total_engagement"] = (
        posts["likes"] + posts["shares"] * 2 + posts["comments"] * 1.5
    )
    return posts.sort_values("total_engagement", ascending=False).head(top_n)
