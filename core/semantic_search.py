"""Semantic search for Kaggle competition code.

Search indexed code snippets by natural language queries.
Uses local embeddings (RTX 5070) + homelab Qdrant storage.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import logging

log = logging.getLogger(__name__)


class CodeSearch:
    """Semantic search for competition code."""

    def __init__(self, collection_name: str = "kaggle_code"):
        """
        Initialize code search.

        Args:
            collection_name: Qdrant collection to search
        """
        self.collection_name = collection_name

    def search(
        self,
        query: str,
        limit: int = 10,
        competition: Optional[str] = None,
        category: Optional[str] = None,
        score_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Search for code snippets by natural language query.

        Args:
            query: Natural language query (e.g., "lag features for time series")
            limit: Maximum number of results
            competition: Filter by competition slug
            category: Filter by category (feature_engineering, model_training, etc.)
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of matching code snippets with scores
        """
        from core.embedder import get_embedder
        from core.memory import qdrant_client
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        # Generate query embedding
        embedder = get_embedder("mpnet")
        query_embedding = embedder.embed(f"Code: {query}", normalize=True)

        # Build filters
        filters = []
        if competition:
            filters.append(FieldCondition(key="competition", match=MatchValue(value=competition)))
        if category:
            filters.append(FieldCondition(key="category", match=MatchValue(value=category)))

        query_filter = Filter(must=filters) if filters else None

        # Search Qdrant
        try:
            client = qdrant_client()
            results = client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
            )

            # Format results
            formatted = []
            for hit in results:
                formatted.append({
                    "code": hit.payload.get("code", ""),
                    "file_path": hit.payload.get("file_path", ""),
                    "competition": hit.payload.get("competition", ""),
                    "type": hit.payload.get("type", ""),
                    "name": hit.payload.get("name", ""),
                    "category": hit.payload.get("category", ""),
                    "score": hit.score,
                })

            return formatted

        except Exception as e:
            log.error(f"Search failed: {e}")
            return []

    def search_similar_code(
        self,
        code: str,
        limit: int = 5,
        competition: Optional[str] = None,
        score_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Find similar code snippets.

        Args:
            code: Code snippet to find similar to
            limit: Maximum number of results
            competition: Filter by competition
            score_threshold: Minimum similarity (higher for code-to-code)

        Returns:
            Similar code snippets
        """
        from core.embedder import get_embedder
        from core.memory import qdrant_client
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        import hashlib

        # Generate embedding for input code
        embedder = get_embedder("mpnet")
        code_embedding = embedder.embed_code(code, add_context=True)

        # Calculate hash to exclude exact match
        code_hash = hashlib.md5(code.encode()).hexdigest()

        # Build filter
        filters = []
        if competition:
            filters.append(FieldCondition(key="competition", match=MatchValue(value=competition)))

        query_filter = Filter(must=filters) if filters else None

        # Search
        try:
            client = qdrant_client()
            results = client.search(
                collection_name=self.collection_name,
                query_vector=code_embedding.tolist(),
                query_filter=query_filter,
                limit=limit + 1,  # Get one extra in case we filter out exact match
                score_threshold=score_threshold,
            )

            # Format and filter out exact match
            formatted = []
            for hit in results:
                if hit.payload.get("hash") == code_hash:
                    continue  # Skip exact match

                formatted.append({
                    "code": hit.payload.get("code", ""),
                    "file_path": hit.payload.get("file_path", ""),
                    "competition": hit.payload.get("competition", ""),
                    "type": hit.payload.get("type", ""),
                    "name": hit.payload.get("name", ""),
                    "category": hit.payload.get("category", ""),
                    "score": hit.score,
                })

                if len(formatted) >= limit:
                    break

            return formatted

        except Exception as e:
            log.error(f"Similar code search failed: {e}")
            return []

    def search_by_category(
        self,
        category: str,
        query: Optional[str] = None,
        limit: int = 10,
        competition: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific category.

        Args:
            category: feature_engineering, model_training, validation, etc.
            query: Optional query to refine results
            limit: Maximum results
            competition: Filter by competition

        Returns:
            Matching code snippets
        """
        if query:
            return self.search(
                query=query,
                limit=limit,
                competition=competition,
                category=category,
                score_threshold=0.3,  # Lower threshold for category-filtered search
            )
        else:
            # Just retrieve by category
            from core.memory import qdrant_client
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue

            filters = [FieldCondition(key="category", match=MatchValue(value=category))]
            if competition:
                filters.append(FieldCondition(key="competition", match=MatchValue(value=competition)))

            try:
                client = qdrant_client()
                results = client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(must=filters),
                    limit=limit,
                )

                formatted = []
                for point in results[0]:
                    formatted.append({
                        "code": point.payload.get("code", ""),
                        "file_path": point.payload.get("file_path", ""),
                        "competition": point.payload.get("competition", ""),
                        "type": point.payload.get("type", ""),
                        "name": point.payload.get("name", ""),
                        "category": point.payload.get("category", ""),
                        "score": 1.0,  # No similarity score for scroll
                    })

                return formatted

            except Exception as e:
                log.error(f"Category search failed: {e}")
                return []


def search_code(
    query: str,
    limit: int = 5,
    competition: Optional[str] = None,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Quick helper to search code."""
    searcher = CodeSearch()
    return searcher.search(
        query=query,
        limit=limit,
        competition=competition,
        category=category,
    )


def find_similar_features(description: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Find similar feature engineering code."""
    searcher = CodeSearch()
    return searcher.search(
        query=description,
        limit=limit,
        category="feature_engineering",
    )


def find_similar_models(description: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Find similar model training code."""
    searcher = CodeSearch()
    return searcher.search(
        query=description,
        limit=limit,
        category="model_training",
    )


# CLI usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m core.semantic_search \"query\"")
        print("Example: python -m core.semantic_search \"lag features for sales\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"Searching for: {query}\n")

    results = search_code(query, limit=5)

    if not results:
        print("No results found.")
    else:
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['name']} ({result['category']}) - Score: {result['score']:.3f}")
            print(f"   Competition: {result['competition']}")
            print(f"   File: {result['file_path']}")
            print(f"   Code preview:\n{result['code'][:300]}...")
            print()
