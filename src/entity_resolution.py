"""Entity mention extraction and resolution.

Strategy (layered, cheapest first):
1. Exact match: case-insensitive substring match against canonical names + aliases
2. Fuzzy match: rapidfuzz token_set_ratio for near-matches
3. Embedding match: sentence-transformer cosine similarity for semantic matches
4. LLM-assisted: Bedrock Claude for ambiguous/low-confidence cases (optional)

Each match produces: {entity_id, mention_text, confidence, resolution_method}
"""

import logging
import re
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer

from src.config import (
    FUZZY_MATCH_THRESHOLD,
    EMBEDDING_SIMILARITY_THRESHOLD,
)
from src.llm import invoke_llm_json

logger = logging.getLogger(__name__)

# Lazy-loaded embedding model
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _build_alias_index(entity_lookup: dict) -> Dict[str, List[tuple]]:
    """Build normalized alias -> (entity_id, original_alias) mapping."""
    index = {}
    for entity_id, info in entity_lookup.items():
        for alias in info["aliases"]:
            key = alias.lower().strip()
            if key not in index:
                index[key] = []
            index[key].append((entity_id, alias))
    return index


def _exact_match(
    text: str, alias_index: dict
) -> List[Dict]:
    """Find exact substring matches (case-insensitive)."""
    matches = []
    text_lower = text.lower()
    for alias_key, entries in alias_index.items():
        if len(alias_key) < 3:
            continue  # Skip very short aliases to avoid false positives
        # Word boundary check to avoid matching inside longer words
        pattern = r"(?<![a-zA-Z])" + re.escape(alias_key) + r"(?![a-zA-Z])"
        if re.search(pattern, text_lower):
            for entity_id, original_alias in entries:
                matches.append(
                    {
                        "entity_id": entity_id,
                        "mention_text": original_alias,
                        "confidence": 0.95,
                        "resolution_method": "exact_alias",
                    }
                )
    # Deduplicate by entity_id, keep highest confidence
    seen = {}
    for m in matches:
        eid = m["entity_id"]
        if eid not in seen or m["confidence"] > seen[eid]["confidence"]:
            seen[eid] = m
    return list(seen.values())


def _fuzzy_match(
    text: str, entity_lookup: dict, threshold: int = FUZZY_MATCH_THRESHOLD
) -> List[Dict]:
    """Fuzzy match against entity names/aliases using rapidfuzz."""
    matches = []
    all_aliases = []
    alias_to_entity = {}
    for entity_id, info in entity_lookup.items():
        for alias in info["aliases"]:
            all_aliases.append(alias)
            alias_to_entity[alias] = entity_id

    # Extract potential entity-like spans (capitalized words, 2-4 word phrases)
    words = text.split()
    candidates = []
    for i in range(len(words)):
        for j in range(i + 1, min(i + 5, len(words) + 1)):
            span = " ".join(words[i:j])
            if len(span) >= 3:
                candidates.append(span)

    for candidate in candidates:
        result = process.extractOne(
            candidate, all_aliases, scorer=fuzz.token_set_ratio
        )
        if result and result[1] >= threshold:
            matched_alias = result[0]
            score = result[1]
            entity_id = alias_to_entity[matched_alias]
            confidence = round(min(score / 100.0, 0.90), 2)  # Cap at 0.90 for fuzzy
            matches.append(
                {
                    "entity_id": entity_id,
                    "mention_text": candidate,
                    "confidence": confidence,
                    "resolution_method": "fuzzy",
                }
            )

    # Deduplicate
    seen = {}
    for m in matches:
        eid = m["entity_id"]
        if eid not in seen or m["confidence"] > seen[eid]["confidence"]:
            seen[eid] = m
    return list(seen.values())


def _embedding_match(
    text: str,
    entity_lookup: dict,
    entity_embeddings: np.ndarray,
    entity_ids_list: List[str],
    threshold: float = EMBEDDING_SIMILARITY_THRESHOLD,
) -> List[Dict]:
    """Semantic similarity match using pre-computed entity embeddings."""
    model = _get_embed_model()
    text_emb = model.encode([text], normalize_embeddings=True)

    # Cosine similarity against pre-computed entity embeddings
    similarities = np.dot(text_emb, entity_embeddings.T)[0]

    matches = []
    for idx, sim in enumerate(similarities):
        if sim >= threshold:
            eid = entity_ids_list[idx]
            matches.append(
                {
                    "entity_id": eid,
                    "mention_text": entity_lookup[eid]["canonical_name"],
                    "confidence": round(float(sim), 2),
                    "resolution_method": "embedding",
                }
            )

    return matches


def _precompute_entity_embeddings(entity_lookup: dict):
    """Pre-compute entity embeddings once for reuse."""
    model = _get_embed_model()
    entity_texts = []
    entity_ids = []
    for entity_id, info in entity_lookup.items():
        desc = f"{info['canonical_name']} ({info['entity_type']}): {', '.join(info['aliases'])}"
        entity_texts.append(desc)
        entity_ids.append(entity_id)
    embeddings = model.encode(entity_texts, normalize_embeddings=True)
    return embeddings, entity_ids


def _batch_embedding_match(
    texts: List[str],
    entity_lookup: dict,
    threshold: float = EMBEDDING_SIMILARITY_THRESHOLD,
) -> List[List[Dict]]:
    """Batch embedding match for multiple texts at once."""
    if not texts:
        return []

    model = _get_embed_model()
    entity_embs, entity_ids = _precompute_entity_embeddings(entity_lookup)

    # Batch encode all texts at once
    text_embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)

    # Batch cosine similarity
    sim_matrix = np.dot(text_embs, entity_embs.T)

    all_matches = []
    for i in range(len(texts)):
        matches = []
        for j, sim in enumerate(sim_matrix[i]):
            if sim >= threshold:
                eid = entity_ids[j]
                matches.append(
                    {
                        "entity_id": eid,
                        "mention_text": entity_lookup[eid]["canonical_name"],
                        "confidence": round(float(sim), 2),
                        "resolution_method": "embedding",
                    }
                )
        all_matches.append(matches)

    return all_matches


def _llm_resolve(
    text: str, entity_lookup: dict, existing_matches: List[Dict]
) -> List[Dict]:
    """Use LLM to resolve ambiguous mentions."""
    entity_list = "\n".join(
        f"- {eid}: {info['canonical_name']} (aliases: {', '.join(info['aliases'])})"
        for eid, info in entity_lookup.items()
    )

    prompt = f"""Analyze this social media post and identify which entities from the list are mentioned or discussed.

Post: "{text}"

Known entities:
{entity_list}

Return JSON array of matches. Each match: {{"entity_id": "...", "mention_text": "the text that refers to the entity", "confidence": 0.0-1.0}}
Only include entities that are clearly referenced. Return empty array [] if none match.
Return ONLY valid JSON, no explanation."""

    result = invoke_llm_json(
        prompt,
        system="You are an entity resolution system. Be precise. Only match entities that are clearly referenced in the text.",
    )

    if result is None:
        return []

    if isinstance(result, list):
        matches = []
        for m in result:
            if isinstance(m, dict) and "entity_id" in m:
                if m["entity_id"] in entity_lookup:
                    matches.append(
                        {
                            "entity_id": m["entity_id"],
                            "mention_text": m.get("mention_text", ""),
                            "confidence": min(float(m.get("confidence", 0.8)), 0.95),
                            "resolution_method": "llm_assisted",
                        }
                    )
        return matches
    return []


def resolve_entities_for_post(
    text: str,
    entity_lookup: dict,
    alias_index: dict,
    use_llm: bool = True,
    entity_embeddings: np.ndarray = None,
    entity_ids_list: List[str] = None,
) -> Dict:
    """Resolve entities for a single post using layered strategy.

    Returns:
        {
            "resolved_entities": [...],
            "resolution_method": "exact_alias|fuzzy|embedding|llm_assisted",
            "needs_review": bool
        }
    """
    # Layer 1: Exact match
    matches = _exact_match(text, alias_index)
    if matches:
        return {
            "resolved_entities": matches,
            "resolution_method": "exact_alias",
            "needs_review": False,
        }

    # Layer 2: Fuzzy match
    matches = _fuzzy_match(text, entity_lookup)
    if matches:
        needs_review = any(m["confidence"] < 0.85 for m in matches)
        return {
            "resolved_entities": matches,
            "resolution_method": "fuzzy",
            "needs_review": needs_review,
        }

    # Layer 3: Embedding match (using pre-computed embeddings if available)
    if entity_embeddings is not None and entity_ids_list is not None:
        matches = _embedding_match(text, entity_lookup, entity_embeddings, entity_ids_list)
    else:
        # Fallback: compute on the fly
        embs, eids = _precompute_entity_embeddings(entity_lookup)
        matches = _embedding_match(text, entity_lookup, embs, eids)

    if matches:
        return {
            "resolved_entities": matches,
            "resolution_method": "embedding",
            "needs_review": True,
        }

    # Layer 4: LLM (if enabled)
    if use_llm:
        matches = _llm_resolve(text, entity_lookup, [])
        if matches:
            return {
                "resolved_entities": matches,
                "resolution_method": "llm_assisted",
                "needs_review": True,
            }

    return {
        "resolved_entities": [],
        "resolution_method": "none",
        "needs_review": False,
    }


def resolve_all_posts(
    posts_df: pd.DataFrame,
    entity_lookup: dict,
    use_llm: bool = True,
    text_column: str = "text",
) -> pd.DataFrame:
    """Resolve entities for all posts using optimised batch processing."""
    alias_index = _build_alias_index(entity_lookup)

    # Phase 1: Exact + Fuzzy (fast, no embeddings needed)
    results = [None] * len(posts_df)
    needs_embedding = []  # indices that need embedding fallback

    for idx, (_, row) in enumerate(posts_df.iterrows()):
        text = row.get(text_column, "")
        if not text or not isinstance(text, str):
            results[idx] = {"resolved_entities": [], "resolution_method": "none", "needs_review": False}
            continue

        # Try exact match
        matches = _exact_match(text, alias_index)
        if matches:
            results[idx] = {"resolved_entities": matches, "resolution_method": "exact_alias", "needs_review": False}
            continue

        # Try fuzzy match
        matches = _fuzzy_match(text, entity_lookup)
        if matches:
            needs_review = any(m["confidence"] < 0.85 for m in matches)
            results[idx] = {"resolved_entities": matches, "resolution_method": "fuzzy", "needs_review": needs_review}
            continue

        needs_embedding.append(idx)

    # Phase 2: Batch embedding for unresolved posts
    if needs_embedding:
        texts_for_embedding = [posts_df.iloc[i][text_column] for i in needs_embedding]
        embedding_results = _batch_embedding_match(texts_for_embedding, entity_lookup)

        for i, idx in enumerate(needs_embedding):
            matches = embedding_results[i]
            if matches:
                results[idx] = {"resolved_entities": matches, "resolution_method": "embedding", "needs_review": True}
            else:
                results[idx] = {"resolved_entities": [], "resolution_method": "none", "needs_review": False}

    # Phase 3: LLM for still-unresolved (only if enabled)
    if use_llm:
        for idx in range(len(results)):
            if results[idx]["resolution_method"] == "none" and results[idx]["resolved_entities"] == []:
                text = posts_df.iloc[idx].get(text_column, "")
                if text and isinstance(text, str):
                    matches = _llm_resolve(text, entity_lookup, [])
                    if matches:
                        results[idx] = {"resolved_entities": matches, "resolution_method": "llm_assisted", "needs_review": True}

    posts_df = posts_df.copy()
    posts_df["resolved_entities"] = [r["resolved_entities"] for r in results]
    posts_df["resolution_method"] = [r["resolution_method"] for r in results]
    posts_df["needs_review"] = [r["needs_review"] for r in results]

    return posts_df


def get_posts_for_entity(posts_df: pd.DataFrame, entity_id: str) -> pd.DataFrame:
    """Filter posts that mention a specific entity."""
    mask = posts_df["resolved_entities"].apply(
        lambda ents: any(e["entity_id"] == entity_id for e in ents) if ents else False
    )
    return posts_df[mask].copy()


def get_entity_confidence_stats(posts_df: pd.DataFrame, entity_id: str) -> Dict:
    """Get confidence distribution for entity matches."""
    entity_posts = get_posts_for_entity(posts_df, entity_id)
    if entity_posts.empty:
        return {"total": 0, "high": 0, "medium": 0, "low": 0, "methods": {}}

    confidences = []
    methods = {}
    for _, row in entity_posts.iterrows():
        for ent in row["resolved_entities"]:
            if ent["entity_id"] == entity_id:
                confidences.append(ent["confidence"])
                method = ent["resolution_method"]
                methods[method] = methods.get(method, 0) + 1

    high = sum(1 for c in confidences if c >= 0.9)
    medium = sum(1 for c in confidences if 0.7 <= c < 0.9)
    low = sum(1 for c in confidences if c < 0.7)

    return {
        "total": len(confidences),
        "high": high,
        "medium": medium,
        "low": low,
        "avg_confidence": round(np.mean(confidences), 2) if confidences else 0,
        "methods": methods,
    }
