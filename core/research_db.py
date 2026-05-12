#!/usr/bin/env python3
"""Research Knowledge Database using Qdrant

Separates three types of knowledge:
1. General ML Research (techniques, architectures, papers)
2. Competition-Specific Learnings (Store Sales, Titanic, etc.)
3. Problem-Solving Patterns (what works when, debugging strategies)

Uses embeddings to enable semantic search and knowledge retrieval.
"""
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


@dataclass
class ResearchEntry:
    """A piece of research knowledge."""
    content: str
    category: str  # 'general_ml', 'competition_specific', 'problem_solving'
    source: str  # Paper name, competition, or 'experimentation'
    tags: List[str]
    confidence: float  # 0.0-1.0
    date_added: str
    metadata: Dict


class ResearchDatabase:
    """Qdrant-based research knowledge system."""

    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=qdrant_url)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good embeddings

        # Create collections if they don't exist
        self._init_collections()

    def _init_collections(self):
        """Initialize three separate collections."""
        collections = {
            "general_ml_research": "General ML techniques, architectures, papers",
            "competition_learnings": "Competition-specific patterns and insights",
            "problem_solving": "Debugging, troubleshooting, decision patterns"
        }

        for collection_name, description in collections.items():
            try:
                self.client.get_collection(collection_name)
                print(f"✓ Collection '{collection_name}' exists")
            except:
                # Create collection
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                print(f"✓ Created collection '{collection_name}'")

    def add_research(self, entry: ResearchEntry):
        """Add research entry to appropriate collection."""
        # Map category to collection
        collection_map = {
            "general_ml": "general_ml_research",
            "competition_specific": "competition_learnings",
            "problem_solving": "problem_solving"
        }

        collection_name = collection_map.get(entry.category)
        if not collection_name:
            raise ValueError(f"Invalid category: {entry.category}")

        # Generate embedding
        vector = self.encoder.encode(entry.content).tolist()

        # Create point
        point = PointStruct(
            id=abs(hash(entry.content + entry.date_added)) % (10**10),  # Positive int < 10B
            vector=vector,
            payload={
                "content": entry.content,
                "source": entry.source,
                "tags": entry.tags,
                "confidence": entry.confidence,
                "date_added": entry.date_added,
                "metadata": entry.metadata
            }
        )

        # Upsert
        self.client.upsert(
            collection_name=collection_name,
            points=[point]
        )

    def search(self, query: str, category: str = "all", limit: int = 5) -> List[Dict]:
        """Semantic search across research knowledge."""
        # Map category to collections
        if category == "all":
            collections = ["general_ml_research", "competition_learnings", "problem_solving"]
        elif category == "general_ml":
            collections = ["general_ml_research"]
        elif category == "competition_specific":
            collections = ["competition_learnings"]
        elif category == "problem_solving":
            collections = ["problem_solving"]
        else:
            raise ValueError(f"Invalid category: {category}")

        # Generate query embedding
        query_vector = self.encoder.encode(query).tolist()

        # Search all relevant collections
        all_results = []
        for collection_name in collections:
            from qdrant_client.models import SearchRequest
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )

            for result in results:
                all_results.append({
                    "collection": collection_name,
                    "score": result.score,
                    "content": result.payload["content"],
                    "source": result.payload["source"],
                    "tags": result.payload["tags"],
                    "confidence": result.payload["confidence"],
                    "metadata": result.payload.get("metadata", {})
                })

        # Sort by relevance score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:limit]

    def add_bert_research(self):
        """Add research about BERT for tabular data."""
        bert_entry = ResearchEntry(
            content="""BERT for Tabular Data (TabBERT/SAINT):

ADVANTAGES over traditional models:
1. Data Efficiency: Works well with smaller datasets (10K-100K rows)
   - Pretrained on large corpus, transfers knowledge
   - Traditional GBMs need 100K+ rows to be competitive

2. Faster Iteration:
   - Can fine-tune in minutes vs hours of hyperparameter search
   - Less sensitive to hyperparameters than GBMs

3. Handles Missing Data:
   - Uses attention mechanism to learn missingness patterns
   - No need for manual imputation strategies

4. Categorical Embeddings:
   - Learns semantic relationships between categories
   - Better than one-hot or label encoding

DISADVANTAGES:
1. Slower inference than GBMs (10-100x)
2. Harder to interpret (black box)
3. Requires GPU for reasonable speed
4. More complex to deploy

WHEN TO USE BERT:
- Small datasets (<50K rows)
- Many categorical features
- Need fast experimentation (limited compute for hyperparameter tuning)
- Missing data is prevalent
- Semantic relationships important

WHEN TO USE GBMs (LightGBM/XGBoost):
- Large datasets (>100K rows)
- Production systems (fast inference critical)
- Need interpretability (feature importance)
- Tabular data with primarily numeric features
- Competition leaderboards (GBMs still dominate)

STORE SALES CONTEXT:
- 3M training rows → GBMs preferred
- Simple numeric features (lags, rolling means) → GBMs excel
- Speed matters for iteration → GBMs faster
- Interpretability valuable → GBMs better
""",
            category="general_ml",
            source="TabBERT (AAAI 2020), SAINT (NeurIPS 2021)",
            tags=["BERT", "tabular", "transformers", "data_efficiency", "time_series"],
            confidence=0.9,
            date_added=datetime.now().isoformat(),
            metadata={
                "papers": ["TabBERT", "SAINT", "TabNet"],
                "use_cases": ["small_data", "categorical_heavy", "missing_data"]
            }
        )
        self.add_research(bert_entry)

    def add_cnn_research(self):
        """Add research about CNNs for time series."""
        cnn_entry = ResearchEntry(
            content="""CNNs for Time Series (Temporal Convolutional Networks - TCNs):

ADVANTAGES:
1. Capture Local Patterns:
   - 1D convolutions detect patterns in sequences
   - Good for seasonality, trends, cycles

2. Parallel Processing:
   - Unlike RNNs, can process entire sequence in parallel
   - Much faster training than LSTM/GRU

3. Long-Range Dependencies:
   - Dilated convolutions capture long-term patterns
   - Better than basic RNNs, competitive with Transformers

4. Less Overfitting:
   - Fewer parameters than Transformers
   - Regularization through convolution sharing

DISADVANTAGES:
1. Fixed receptive field (need to tune kernel sizes)
2. Less flexible than attention mechanisms
3. Still needs more data than GBMs (50K+ rows)
4. Harder to interpret than tree models

WHEN TO USE CNNs/TCNs:
- Time series with clear patterns (seasonality, trends)
- Medium datasets (50K-500K rows)
- Need faster training than RNNs
- Pattern recognition more important than absolute values

STORE SALES CONTEXT:
- Daily time series with weekly/monthly patterns → Could work
- BUT: 16-day forecast too short to benefit from deep learning
- GBMs with lag/rolling features capture patterns well
- TCN would be overkill for this problem
""",
            category="general_ml",
            source="TCN (2018), WaveNet (2016)",
            tags=["CNN", "time_series", "TCN", "deep_learning"],
            confidence=0.85,
            date_added=datetime.now().isoformat(),
            metadata={
                "papers": ["Temporal Convolutional Networks", "WaveNet"],
                "use_cases": ["time_series", "sequence_modeling", "audio"]
            }
        )
        self.add_research(cnn_entry)

    def add_store_sales_learnings(self):
        """Add competition-specific learnings from Store Sales."""
        learnings = [
            ResearchEntry(
                content="Simple features (12) outperform complex features (14-27). v19's lags + rolling means capture all useful signal. Adding store-family-dow, automated features, or temporal features consistently hurts performance (3-23% worse).",
                category="competition_specific",
                source="store-sales-time-series-forecasting",
                tags=["feature_engineering", "simplicity", "validated"],
                confidence=0.95,
                date_added=datetime.now().isoformat(),
                metadata={"experiments": ["v19", "v56", "v57", "v58"], "impact": "high"}
            ),
            ResearchEntry(
                content="More historical data is better. Training on 2016-2017 only (66% data reduction) performed 3.4% worse than full 2013-2017 history. Sales patterns from 2013-2015 are still predictive for 2017.",
                category="competition_specific",
                source="store-sales-time-series-forecasting",
                tags=["data_volume", "time_series", "validated"],
                confidence=0.9,
                date_added=datetime.now().isoformat(),
                metadata={"experiment": "v57", "data_years": "2013-2017 vs 2016-2017"}
            ),
            ResearchEntry(
                content="Recursive forecasting doesn't help for short horizons (16 days). v55 predictions had 0.99 correlation with batch predictions, suggesting lag features calculated recursively vs with means produce similar results.",
                category="competition_specific",
                source="store-sales-time-series-forecasting",
                tags=["forecasting_strategy", "recursive", "validated"],
                confidence=0.85,
                date_added=datetime.now().isoformat(),
                metadata={"experiment": "v55", "correlation": 0.99, "horizon_days": 16}
            ),
            ResearchEntry(
                content="Ridge stacking with intercept provides 5.8% improvement over simple weighted averaging. The intercept term (-0.4937) corrects systematic bias in base model predictions. Always use fit_intercept=True for meta-learning.",
                category="competition_specific",
                source="store-sales-time-series-forecasting",
                tags=["ensembling", "ridge", "intercept", "critical"],
                confidence=0.95,
                date_added=datetime.now().isoformat(),
                metadata={"experiment": "v50/v54", "improvement": "5.8%", "technique": "stacking"}
            )
        ]

        for entry in learnings:
            self.add_research(entry)

    def add_problem_solving_patterns(self):
        """Add general problem-solving insights."""
        patterns = [
            ResearchEntry(
                content="When all improvements fail (0/4 experiments successful), the baseline is likely already optimal for the problem. Adding complexity → overfitting. Solution: Accept baseline, try fundamentally different approach, or move to next problem.",
                category="problem_solving",
                source="experimentation",
                tags=["optimization", "baseline", "decision_making"],
                confidence=0.9,
                date_added=datetime.now().isoformat(),
                metadata={"pattern": "diminishing_returns", "action": "stop_optimizing"}
            ),
            ResearchEntry(
                content="Research-backed techniques don't always work in practice. Validate on YOUR data, not papers. 4/4 research techniques failed on Store Sales despite strong theoretical support. Data trumps theory.",
                category="problem_solving",
                source="experimentation",
                tags=["research", "validation", "pragmatism"],
                confidence=0.95,
                date_added=datetime.now().isoformat(),
                metadata={"lesson": "validate_empirically", "context": "store_sales"}
            ),
            ResearchEntry(
                content="Simple A/B testing > complex optimization when baseline is strong. Test one change at a time. If change doesn't improve, revert immediately. Don't combine multiple changes that individually failed.",
                category="problem_solving",
                source="experimentation",
                tags=["ab_testing", "methodology", "iteration"],
                confidence=0.9,
                date_added=datetime.now().isoformat(),
                metadata={"strategy": "incremental", "principle": "occams_razor"}
            )
        ]

        for entry in patterns:
            self.add_research(entry)


def initialize_research_db():
    """Initialize research database with foundational knowledge."""
    print("Initializing Research Database...")
    print("="*70)

    db = ResearchDatabase()

    print("\nAdding BERT research...")
    db.add_bert_research()

    print("Adding CNN/TCN research...")
    db.add_cnn_research()

    print("Adding Store Sales learnings...")
    db.add_store_sales_learnings()

    print("Adding problem-solving patterns...")
    db.add_problem_solving_patterns()

    print("\n✓ Research database initialized")
    print("="*70)

    # Test search
    print("\nTesting semantic search:")
    print("\nQuery: 'Should I use BERT for tabular data with 100K rows?'")
    results = db.search("Should I use BERT for tabular data with 100K rows?", limit=2)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Source: {result['source']}")
        print(f"   Tags: {', '.join(result['tags'])}")
        print(f"   Content: {result['content'][:200]}...")

    return db


if __name__ == "__main__":
    db = initialize_research_db()
    print("\n✓ Research database ready for use")
