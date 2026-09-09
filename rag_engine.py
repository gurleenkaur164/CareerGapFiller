"""
RAG Engine for Career Gap-Filler
─────────────────────────────────
Lightweight TF-IDF embeddings (scikit-learn) + in-memory vector search.

Pipeline:
  missing skill → create query → TF-IDF embed → cosine similarity → top 3 resources

Uses scikit-learn instead of sentence-transformers + ChromaDB for maximum
compatibility (no PyTorch / CUDA / DLL dependencies).
"""

import json
import logging
import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
SEED_FILE = os.path.join(os.path.dirname(__file__), "resources_seed.json")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "tfidf_cache")
VECTORIZER_PATH = os.path.join(CACHE_DIR, "vectorizer.pkl")
MATRIX_PATH = os.path.join(CACHE_DIR, "tfidf_matrix.pkl")

# Module-level singletons
_vectorizer: TfidfVectorizer | None = None
_tfidf_matrix = None  # sparse matrix of shape (n_resources, n_features)
_resources: list | None = None


def _resource_document(resource: dict) -> str:
    """Build the text that gets embedded for a learning resource."""
    skills_text = " ".join(resource.get("skills", []))
    return (
        f"{resource.get('title', '')}. "
        f"{resource.get('description', '')}. "
        f"Skills: {skills_text}. "
        f"Difficulty: {resource.get('difficulty', '')}. "
        f"Type: {resource.get('type', '')}."
    )


def create_query_for_skill(skill_entry):
    """
    Create a retrieval query for one missing skill.

    Args:
        skill_entry: Skill name string, or dict with skill/importance/reason.

    Returns:
        (query_text, skill_name)
    """
    if isinstance(skill_entry, str):
        skill_name = skill_entry.strip()
        importance = ""
        reason = ""
    else:
        skill_name = (skill_entry.get("skill") or "").strip()
        importance = skill_entry.get("importance", "") or ""
        reason = skill_entry.get("reason", "") or ""

    query_parts = [
        f"learn {skill_name} tutorial course video documentation practice",
        skill_name,
    ]
    if importance:
        query_parts.append(importance)
    if reason:
        query_parts.append(reason)

    return " ".join(query_parts), skill_name


def init_vector_store(force_rebuild=False):
    """
    Initialize the TF-IDF vector store from resources_seed.json.

    On first run (or if force_rebuild=True / seed size changed):
      - Loads resources_seed.json
      - Builds a TF-IDF matrix over all resource documents
      - Caches the vectorizer + matrix to disk for fast reloads

    On subsequent runs:
      - Loads from cache if seed hasn't changed.

    Returns True if initialization succeeded.
    """
    global _vectorizer, _tfidf_matrix, _resources

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load seed resources
    if not os.path.exists(SEED_FILE):
        logger.warning("Seed file not found at %s. Vector store will be empty.", SEED_FILE)
        _resources = []
        return False

    with open(SEED_FILE, "r", encoding="utf-8") as f:
        _resources = json.load(f)

    # Check cache validity
    if (
        not force_rebuild
        and os.path.exists(VECTORIZER_PATH)
        and os.path.exists(MATRIX_PATH)
    ):
        try:
            with open(VECTORIZER_PATH, "rb") as f:
                cached_vectorizer = pickle.load(f)
            with open(MATRIX_PATH, "rb") as f:
                cached_matrix = pickle.load(f)

            if cached_matrix.shape[0] == len(_resources):
                _vectorizer = cached_vectorizer
                _tfidf_matrix = cached_matrix
                logger.info(
                    "TF-IDF cache loaded from disk with %d resources.",
                    len(_resources),
                )
                return True
            else:
                logger.info(
                    "Cache has %d rows but seed has %d resources. Rebuilding...",
                    cached_matrix.shape[0],
                    len(_resources),
                )
        except Exception as e:
            logger.warning("Cache load failed (%s). Rebuilding...", e)

    # Build TF-IDF matrix from scratch
    logger.info("Building TF-IDF matrix for %d resources...", len(_resources))
    documents = [_resource_document(r) for r in _resources]

    _vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),  # unigrams + bigrams for better matching
        sublinear_tf=True,
    )
    _tfidf_matrix = _vectorizer.fit_transform(documents)

    # Cache to disk
    try:
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(_vectorizer, f)
        with open(MATRIX_PATH, "wb") as f:
            pickle.dump(_tfidf_matrix, f)
        logger.info("TF-IDF cache saved to %s", CACHE_DIR)
    except Exception as e:
        logger.warning("Could not save TF-IDF cache: %s", e)

    logger.info(
        "Indexed %d resources (TF-IDF matrix shape: %s)",
        len(_resources),
        _tfidf_matrix.shape,
    )
    return True


def retrieve_resources(missing_skills, top_k=3):
    """
    Retrieve the most relevant learning resources for each missing skill.

    For each skill: create a query → TF-IDF transform →
    cosine similarity against all resources → return top_k.

    Args:
        missing_skills: List of skill dicts, each with at least a "skill" key.
                        Example: [{"skill": "Docker", "importance": "Critical", ...}]
        top_k: Number of resources to retrieve per skill (default 3).

    Returns:
        Dict mapping skill name → list of resource dicts.
    """
    global _vectorizer, _tfidf_matrix, _resources

    if _vectorizer is None or _tfidf_matrix is None:
        init_vector_store()

    if not _resources or _vectorizer is None or _tfidf_matrix is None:
        logger.warning("Vector store is empty. No resources to retrieve.")
        return {}

    results = {}
    seen_urls = set()

    for skill_entry in missing_skills:
        query, skill_name = create_query_for_skill(skill_entry)
        if not skill_name:
            continue

        logger.info("RAG query for '%s': %s", skill_name, query)

        # Transform the query with the fitted vectorizer
        query_vec = _vectorizer.transform([query])

        # Compute cosine similarity against all resources
        similarities = cosine_similarity(query_vec, _tfidf_matrix).flatten()

        # Get top indices sorted by similarity (descending)
        top_indices = np.argsort(similarities)[::-1]

        skill_resources = []
        for idx in top_indices:
            sim_score = float(similarities[idx])
            if sim_score <= 0:
                break  # no more relevant results

            resource = _resources[idx]
            url = resource.get("url", "")
            if not url or url in seen_urls:
                continue

            seen_urls.add(url)
            skill_resources.append({
                "title": resource.get("title", ""),
                "url": url,
                "type": resource.get("type", "Link"),
                "description": resource.get("description", ""),
                "difficulty": resource.get("difficulty", ""),
                "distance": round(1.0 - sim_score, 4),  # convert similarity to distance
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
