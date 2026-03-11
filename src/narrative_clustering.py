"""Narrative clustering for entity-specific posts.

Strategy:
1. Filter posts for the selected entity
2. Generate sentence embeddings for each post
3. Cluster using HDBSCAN (handles noise, no need to pre-specify k)
4. Generate narrative titles/summaries (LLM or extractive fallback)
5. Assign risk taxonomy labels
"""

import logging
from typing import List, Dict, Optional
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.config import MIN_CLUSTER_SIZE, MAX_NARRATIVES, RISK_TAXONOMY
from src.llm import invoke_llm_json, invoke_llm

logger = logging.getLogger(__name__)

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _cluster_posts(embeddings: np.ndarray, min_size: int = MIN_CLUSTER_SIZE) -> np.ndarray:
    """Cluster embeddings using Agglomerative Clustering with automatic n_clusters."""
    n_samples = len(embeddings)
    if n_samples < min_size:
        return np.zeros(n_samples, dtype=int)

    # Use agglomerative clustering with distance threshold
    # This automatically determines number of clusters
    try:
        sim_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - sim_matrix
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = np.clip(distance_matrix, 0, None)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.65,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(distance_matrix)

        # Merge small clusters into noise (-1)
        counts = Counter(labels)
        for label, count in counts.items():
            if count < min_size:
                labels[labels == label] = -1

        # Limit to MAX_NARRATIVES largest clusters
        valid_labels = [l for l in set(labels) if l != -1]
        if len(valid_labels) > MAX_NARRATIVES:
            top_labels = sorted(valid_labels, key=lambda l: -sum(labels == l))[:MAX_NARRATIVES]
            for i in range(len(labels)):
                if labels[i] not in top_labels and labels[i] != -1:
                    labels[i] = -1

        return labels

    except Exception as e:
        logger.warning(f"Clustering failed: {e}, falling back to single cluster")
        return np.zeros(n_samples, dtype=int)


def _extractive_title(posts: pd.DataFrame) -> str:
    """Generate a title from the most representative post using TF-IDF-like approach."""
    texts = posts["text"].tolist()
    if not texts:
        return "Unknown Narrative"

    # Use the shortest post that's still meaningful as a title seed
    # Or use the most engaged post
    if "likes" in posts.columns:
        top_post = posts.sort_values(
            by=["likes", "shares", "comments"], ascending=False
        ).iloc[0]["text"]
    else:
        top_post = texts[0]

    # Truncate to a reasonable title length
    title = top_post[:120].strip()
    if len(top_post) > 120:
        title = title.rsplit(" ", 1)[0] + "..."
    return title


def _extractive_summary(posts: pd.DataFrame) -> str:
    """Generate a summary from cluster posts."""
    texts = posts["text"].tolist()
    if not texts:
        return "No posts available."

    # Pick top 2 most engaged posts as representative
    if "likes" in posts.columns:
        top = posts.sort_values(
            by=["likes", "shares", "comments"], ascending=False
        ).head(2)
    else:
        top = posts.head(2)

    snippets = []
    for _, row in top.iterrows():
        snippet = row["text"][:200].strip()
        if len(row["text"]) > 200:
            snippet = snippet.rsplit(" ", 1)[0] + "..."
        snippets.append(snippet)

    return " | ".join(snippets)


def _llm_generate_narrative(posts: pd.DataFrame, entity_name: str) -> Optional[Dict]:
    """Use LLM to generate narrative title, summary, and taxonomy labels."""
    sample_texts = posts.head(8)["text"].tolist()
    posts_text = "\n---\n".join(sample_texts)

    taxonomy_str = "\n".join(f"- {k}: {v}" for k, v in RISK_TAXONOMY.items())

    prompt = f"""Analyze these social media posts about {entity_name} that form a narrative cluster.

Posts:
{posts_text}

Risk taxonomy categories:
{taxonomy_str}

Return JSON:
{{
  "title": "short narrative title (max 10 words)",
  "summary": "1-2 sentence grounded summary of the narrative",
  "taxonomy_labels": ["list of matching taxonomy category keys"],
  "key_themes": ["list of 2-4 key themes/topics"]
}}

Be factual and grounded in the post content. Do not speculate."""

    return invoke_llm_json(
        prompt,
        system="You are a narrative analyst. Generate concise, factual narrative descriptions grounded in evidence.",
    )


def _classify_taxonomy_local(posts: pd.DataFrame) -> List[str]:
    """Rule-based taxonomy classification using keyword matching."""
    all_text = " ".join(posts["text"].str.lower().tolist())

    keyword_map = {
        "regulatory_compliance": [
            "fda", "ema", "regulation", "compliance", "fine", "investigation",
            "breach", "violation", "approved", "approval", "recall", "ban",
            "lawsuit", "legal", "court", "ruling", "penalty",
        ],
        "financial_integrity": [
            "fraud", "money laundering", "manipulation", "profit", "revenue",
            "stock", "share price", "investor", "mis-selling", "overcharged",
        ],
        "customer_harm": [
            "side effect", "adverse", "death", "died", "injury", "harm",
            "complaint", "unsafe", "dangerous", "victim", "suffering",
            "patient", "damaged", "disability",
        ],
        "data_cyber": [
            "data breach", "hack", "leak", "ransomware", "cybersecurity",
            "privacy", "personal data", "stolen",
        ],
        "operational_resilience": [
            "outage", "shortage", "supply chain", "disruption", "delay",
            "unavailable", "failure", "recall",
        ],
        "executive_misconduct": [
            "ceo", "executive", "scandal", "harassment", "misconduct",
            "fired", "resigned", "corrupt",
        ],
        "misinfo_manipulation": [
            "conspiracy", "fake", "hoax", "misinformation", "propaganda",
            "bot", "coordinated", "astroturf", "shill", "cover-up",
            "coverup", "they don't want you to know", "big pharma",
        ],
    }

    labels = []
    for category, keywords in keyword_map.items():
        if any(kw in all_text for kw in keywords):
            labels.append(category)

    return labels if labels else ["misinfo_manipulation"]  # default if nothing matches


def cluster_narratives(
    entity_posts: pd.DataFrame,
    entity_id: str,
    entity_name: str,
    use_llm: bool = True,
) -> List[Dict]:
    """Cluster posts into narratives for a given entity.

    Returns list of narrative dicts:
    {
        narrative_id, title, summary, post_ids, taxonomy_labels,
        key_themes, post_count, member_posts (DataFrame)
    }
    """
    if entity_posts.empty:
        return []

    model = _get_embed_model()
    texts = entity_posts["text"].fillna("").tolist()
    embeddings = model.encode(texts, show_progress_bar=False)

    labels = _cluster_posts(embeddings)

    narratives = []
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        if label == -1:
            continue  # Skip noise

        mask = labels == label
        cluster_posts = entity_posts.iloc[mask].copy()

        if len(cluster_posts) < MIN_CLUSTER_SIZE:
            continue

        narrative_id = f"{entity_id}_narrative_{label}"

        # Try LLM first, fall back to extractive
        llm_result = None
        if use_llm:
            llm_result = _llm_generate_narrative(cluster_posts, entity_name)

        if llm_result:
            title = llm_result.get("title", _extractive_title(cluster_posts))
            summary = llm_result.get("summary", _extractive_summary(cluster_posts))
            taxonomy_labels = llm_result.get("taxonomy_labels", _classify_taxonomy_local(cluster_posts))
            key_themes = llm_result.get("key_themes", [])
        else:
            title = _extractive_title(cluster_posts)
            summary = _extractive_summary(cluster_posts)
            taxonomy_labels = _classify_taxonomy_local(cluster_posts)
            key_themes = []

        narratives.append(
            {
                "narrative_id": narrative_id,
                "entity_id": entity_id,
                "title": title,
                "summary": summary,
                "post_ids": cluster_posts["post_id"].tolist(),
                "post_count": len(cluster_posts),
                "taxonomy_labels": taxonomy_labels,
                "key_themes": key_themes,
                "member_posts": cluster_posts,
            }
        )

    # Also add unclustered posts as "Other / Miscellaneous" if there are enough
    noise_mask = labels == -1
    noise_posts = entity_posts.iloc[noise_mask]
    if len(noise_posts) >= 2:
        narratives.append(
            {
                "narrative_id": f"{entity_id}_narrative_other",
                "entity_id": entity_id,
                "title": "Other / Miscellaneous mentions",
                "summary": f"{len(noise_posts)} posts that don't form a clear narrative cluster.",
                "post_ids": noise_posts["post_id"].tolist(),
                "post_count": len(noise_posts),
                "taxonomy_labels": _classify_taxonomy_local(noise_posts),
                "key_themes": [],
                "member_posts": noise_posts,
            }
        )

    return narratives
