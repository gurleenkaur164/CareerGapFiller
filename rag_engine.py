"""
RAG Engine for Career Gap-Filler
─────────────────────────────────
Uses scikit-learn TF-IDF for local embeddings and a simple
numpy-based vector store (persisted as pickle).

Zero-cost, pure-Python: no C++ build tools, no API calls.
Works on any Python version without compilation.
"""

import json
import os
import pickle
import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
SEED_FILE = os.path.join(os.path.dirname(__file__), "resources_seed.json")
STORE_DIR = os.path.join(os.path.dirname(__file__), "vector_store")
INDEX_FILE = os.path.join(STORE_DIR, "tfidf_index.pkl")

# Module-level singletons
_vectorizer = None
_tfidf_matrix = None
_resources = None


def init_vector_store(force_rebuild=False):
    """
    Initialize the TF-IDF vector store.

    On first run (or if force_rebuild=True):
      - Loads resources_seed.json
      - Builds a TF-IDF matrix from resource descriptions
      - Persists the vectorizer + matrix to ./vector_store/

    On subsequent runs:
      - Loads the persisted index (instant).

    Returns True if initialization succeeded.
    """
    global _vectorizer, _tfidf_matrix, _resources

    os.makedirs(STORE_DIR, exist_ok=True)

    # ── Try loading existing index ──
    if not force_rebuild and os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, "rb") as f:
                data = pickle.load(f)
            _vectorizer = data["vectorizer"]
            _tfidf_matrix = data["tfidf_matrix"]
            _resources = data["resources"]
            logger.info(
                "Vector store loaded from disk with %d resources.", len(_resources)
            )
            return True
        except Exception as e:
            logger.warning("Failed to load existing index: %s. Rebuilding...", e)

    # ── Build from seed data ──
    if not os.path.exists(SEED_FILE):
        logger.warning("Seed file not found at %s. Vector store will be empty.", SEED_FILE)
        _resources = []
        return False

    with open(SEED_FILE, "r", encoding="utf-8") as f:
        _resources = json.load(f)

    logger.info("Building TF-IDF index for %d resources...", len(_resources))

    # Create rich text documents for TF-IDF
    documents = []
    for resource in _resources:
        skills_text = " ".join(resource.get("skills", []))
        doc_text = (
            f"{resource['title']} "
            f"{resource.get('description', '')} "
            f"{skills_text} "
            f"{resource.get('difficulty', '')} "
            f"{resource.get('type', '')} "
            # Repeat skills for higher weight in TF-IDF
            f"{skills_text} {skills_text}"
        )
        documents.append(doc_text.lower())

    # Build TF-IDF matrix
    _vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),  # unigrams + bigrams for better matching
        sublinear_tf=True,
    )
    _tfidf_matrix = _vectorizer.fit_transform(documents)

    # Persist to disk
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(
            {
                "vectorizer": _vectorizer,
                "tfidf_matrix": _tfidf_matrix,
                "resources": _resources,
            },
            f,
        )

    logger.info(
        "✓ Indexed %d resources into vector store at %s",
        len(_resources),
        STORE_DIR,
    )
    return True


def retrieve_resources(missing_skills, top_k=3):
    """
    Retrieve the most relevant learning resources for each missing skill.

    Uses TF-IDF + cosine similarity for semantic matching.

    Args:
        missing_skills: List of skill dicts, each with at least a "skill" key.
                        Example: [{"skill": "Docker", "importance": "Critical", ...}]
        top_k: Number of resources to retrieve per skill (default 3).

    Returns:
        Dict mapping skill name → list of resource dicts:
        {
            "Docker": [
                {"title": "...", "url": "...", "type": "...", "description": "..."},
                ...
            ]
        }
    """
    global _vectorizer, _tfidf_matrix, _resources

    if _vectorizer is None or _resources is None:
        init_vector_store()

    if not _resources:
        logger.warning("Vector store is empty. No resources to retrieve.")
        return {}

    results = {}
    seen_urls = set()  # Global deduplication across skills

    for skill_entry in missing_skills:
        skill_name = skill_entry if isinstance(skill_entry, str) else skill_entry.get("skill", "")
        if not skill_name:
            continue

        # Create a semantic query for this skill
        importance = ""
        if isinstance(skill_entry, dict):
            importance = skill_entry.get("importance", "")
            reason = skill_entry.get("reason", "")

        query = f"learn {skill_name}"
        if importance:
            query += f" {importance}"
        if importance:
            query += f" {reason}"

        # Transform query to TF-IDF vector
        query_vec = _vectorizer.transform([query.lower()])

        # Compute cosine similarity against all resources
        similarities = cosine_similarity(query_vec, _tfidf_matrix).flatten()

        # Get top indices sorted by similarity (descending)
        top_indices = np.argsort(similarities)[::-1]

        skill_resources = []
        for idx in top_indices:
            if similarities[idx] <= 0:
                break  # No more relevant results

            resource = _resources[idx]
            url = resource.get("url", "")

            if url in seen_urls:
                continue  # Skip duplicates across skills

            seen_urls.add(url)
            skill_resources.append({
                "title": resource.get("title", ""),
                "url": url,
                "type": resource.get("type", "Link"),
                "description": resource.get("description", ""),
                "difficulty": resource.get("difficulty", ""),
            })

            if len(skill_resources) >= top_k:
                break

        results[skill_name] = skill_resources

    total = sum(len(v) for v in results.values())
    logger.info("Retrieved %d total resources for %d skills.", total, len(results))
    return results


def get_flat_resources_for_prompt(retrieved_resources):
    """
    Flatten the per-skill retrieved resources into a formatted string
    suitable for inclusion in an LLM prompt.

    Args:
        retrieved_resources: Dict from retrieve_resources().

    Returns:
        A formatted string listing all resources grouped by skill.
    """
    if not retrieved_resources:
        return "No resources found in the knowledge base."

    lines = []
    for skill, resources in retrieved_resources.items():
        lines.append(f"\n### Resources for: {skill}")
        for r in resources:
            lines.append(
                f"  - [{r['type']}] {r['title']}: {r['url']}"
                f"\n    {r['description']}"
            )

    return "\n".join(lines)
