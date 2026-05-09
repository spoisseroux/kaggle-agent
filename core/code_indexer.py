"""Index Kaggle competition code for semantic search.

Scans past competition directories, extracts Python code,
generates embeddings, and stores in Qdrant for fast retrieval.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Dict, Any
import logging
import hashlib

log = logging.getLogger(__name__)

# Code patterns to extract
CODE_PATTERNS = {
    "function": r"def\s+(\w+)\s*\([^)]*\):",
    "class": r"class\s+(\w+)(?:\([^)]*\))?:",
    "feature_engineering": r"(Lag_\d+|Roll_\w+|shift\(|rolling\(|ewm\()",
    "model_training": r"(XGBRegressor|LGBMRegressor|CatBoostRegressor|fit\()",
}


class CodeIndexer:
    """Index competition code for semantic search."""

    def __init__(self, competitions_dir: Path | str = None):
        """
        Initialize indexer.

        Args:
            competitions_dir: Path to competitions directory
                            (default: repo_root/competitions)
        """
        if competitions_dir is None:
            repo_root = Path(__file__).parent.parent
            competitions_dir = repo_root / "competitions"

        self.competitions_dir = Path(competitions_dir)
        self.indexed_files: List[Dict[str, Any]] = []

    def scan_competition(self, competition_slug: str) -> List[Dict[str, Any]]:
        """
        Scan a single competition directory for code.

        Args:
            competition_slug: Competition slug (e.g., "store-sales-time-series-forecasting")

        Returns:
            List of code snippets with metadata
        """
        comp_dir = self.competitions_dir / "active" / competition_slug

        if not comp_dir.exists():
            # Try completed directory
            comp_dir = self.competitions_dir / "completed" / competition_slug

        if not comp_dir.exists():
            log.warning(f"Competition directory not found: {competition_slug}")
            return []

        snippets = []

        # Scan for Python files
        for py_file in comp_dir.rglob("*.py"):
            # Skip __pycache__, .ipynb_checkpoints, etc.
            if any(p in str(py_file) for p in ["__pycache__", ".ipynb_checkpoints", "venv", ".git"]):
                continue

            try:
                snippets.extend(self._extract_snippets(py_file, competition_slug))
            except Exception as e:
                log.warning(f"Failed to extract from {py_file}: {e}")

        log.info(f"Extracted {len(snippets)} snippets from {competition_slug}")
        return snippets

    def _extract_snippets(self, file_path: Path, competition_slug: str) -> List[Dict[str, Any]]:
        """Extract code snippets from a Python file."""
        snippets = []

        try:
            code = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.warning(f"Failed to decode {file_path}")
            return []

        # Extract functions
        for match in re.finditer(r"(def\s+\w+\s*\([^)]*\):.*?)(?=\ndef\s|\nclass\s|\Z)", code, re.DOTALL):
            func_code = match.group(1).strip()

            # Skip very short or very long functions
            if len(func_code) < 50 or len(func_code) > 5000:
                continue

            # Extract function name
            name_match = re.search(r"def\s+(\w+)", func_code)
            if not name_match:
                continue

            func_name = name_match.group(1)

            # Determine snippet type
            snippet_type = self._classify_snippet(func_code)

            snippets.append({
                "code": func_code,
                "file_path": str(file_path.relative_to(self.competitions_dir)),
                "competition": competition_slug,
                "type": "function",
                "name": func_name,
                "category": snippet_type,
                "hash": hashlib.md5(func_code.encode()).hexdigest(),
            })

        # Also extract class definitions
        for match in re.finditer(r"(class\s+\w+(?:\([^)]*\))?:.*?)(?=\nclass\s|\ndef\s|\Z)", code, re.DOTALL):
            class_code = match.group(1).strip()

            if len(class_code) < 50 or len(class_code) > 5000:
                continue

            name_match = re.search(r"class\s+(\w+)", class_code)
            if not name_match:
                continue

            class_name = name_match.group(1)
            snippet_type = self._classify_snippet(class_code)

            snippets.append({
                "code": class_code,
                "file_path": str(file_path.relative_to(self.competitions_dir)),
                "competition": competition_slug,
                "type": "class",
                "name": class_name,
                "category": snippet_type,
                "hash": hashlib.md5(class_code.encode()).hexdigest(),
            })

        return snippets

    def _classify_snippet(self, code: str) -> str:
        """Classify code snippet by purpose."""
        code_lower = code.lower()

        if any(kw in code_lower for kw in ["lag", "shift", "rolling", "ewm", "feature"]):
            return "feature_engineering"
        elif any(kw in code_lower for kw in ["xgb", "lgbm", "catboost", "randomforest", "fit", "train"]):
            return "model_training"
        elif any(kw in code_lower for kw in ["cv", "cross_val", "timeseriessplit", "kfold"]):
            return "validation"
        elif any(kw in code_lower for kw in ["optuna", "hyperopt", "gridsearch", "tune"]):
            return "hyperparameter_tuning"
        elif any(kw in code_lower for kw in ["plot", "visualize", "matplotlib", "seaborn"]):
            return "visualization"
        else:
            return "utility"

    def scan_all_competitions(self) -> List[Dict[str, Any]]:
        """Scan all competitions in the directory."""
        all_snippets = []

        # Scan active competitions
        active_dir = self.competitions_dir / "active"
        if active_dir.exists():
            for comp_dir in active_dir.iterdir():
                if comp_dir.is_dir():
                    snippets = self.scan_competition(comp_dir.name)
                    all_snippets.extend(snippets)

        # Scan completed competitions
        completed_dir = self.competitions_dir / "completed"
        if completed_dir.exists():
            for comp_dir in completed_dir.iterdir():
                if comp_dir.is_dir():
                    snippets = self.scan_competition(comp_dir.name)
                    all_snippets.extend(snippets)

        log.info(f"Total snippets extracted: {len(all_snippets)}")
        return all_snippets

    def index_to_qdrant(
        self,
        snippets: List[Dict[str, Any]],
        collection_name: str = "kaggle_code",
        batch_size: int = 32,
    ) -> int:
        """
        Generate embeddings and store in Qdrant.

        Args:
            snippets: Code snippets to index
            collection_name: Qdrant collection name
            batch_size: Batch size for embedding generation

        Returns:
            Number of snippets indexed
        """
        from core.embedder import get_embedder
        from core.memory import qdrant_client, ensure_collections
        from qdrant_client.http.models import PointStruct
        import uuid

        # Ensure collection exists
        ensure_collections({collection_name: 1536})

        # Get embedder
        embedder = get_embedder("mpnet")

        # Deduplicate by hash
        seen_hashes = set()
        unique_snippets = []
        for snippet in snippets:
            if snippet["hash"] not in seen_hashes:
                seen_hashes.add(snippet["hash"])
                unique_snippets.append(snippet)

        log.info(f"Indexing {len(unique_snippets)} unique snippets (removed {len(snippets) - len(unique_snippets)} duplicates)")

        # Batch process
        client = qdrant_client()
        indexed_count = 0

        for i in range(0, len(unique_snippets), batch_size):
            batch = unique_snippets[i:i + batch_size]

            # Generate embeddings
            codes = [s["code"] for s in batch]
            embeddings = embedder.embed_code(codes, add_context=True)

            # Create points
            points = []
            for snippet, embedding in zip(batch, embeddings):
                point_id = str(uuid.uuid4())
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "code": snippet["code"],
                        "file_path": snippet["file_path"],
                        "competition": snippet["competition"],
                        "type": snippet["type"],
                        "name": snippet["name"],
                        "category": snippet["category"],
                        "hash": snippet["hash"],
                    }
                ))

            # Upsert to Qdrant
            client.upsert(collection_name=collection_name, points=points)
            indexed_count += len(points)

            if (i + batch_size) % 100 == 0:
                log.info(f"Indexed {indexed_count}/{len(unique_snippets)} snippets...")

        log.info(f"✅ Indexed {indexed_count} snippets to {collection_name}")
        return indexed_count


# CLI usage
if __name__ == "__main__":
    import sys

    indexer = CodeIndexer()

    if len(sys.argv) > 1:
        # Index specific competition
        comp_slug = sys.argv[1]
        snippets = indexer.scan_competition(comp_slug)
    else:
        # Index all competitions
        snippets = indexer.scan_all_competitions()

    print(f"\nExtracted {len(snippets)} code snippets")

    # Show sample
    if snippets:
        print("\nSample snippet:")
        sample = snippets[0]
        print(f"  Competition: {sample['competition']}")
        print(f"  File: {sample['file_path']}")
        print(f"  Type: {sample['type']} ({sample['category']})")
        print(f"  Name: {sample['name']}")
        print(f"  Code preview: {sample['code'][:200]}...")

    # Ask to index
    if snippets:
        answer = input(f"\nIndex {len(snippets)} snippets to Qdrant? (y/n): ")
        if answer.lower() == 'y':
            count = indexer.index_to_qdrant(snippets)
            print(f"✅ Indexed {count} snippets")
