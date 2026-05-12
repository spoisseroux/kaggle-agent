#!/usr/bin/env python3
"""Hypothesis Database - Learn from Experiment History

Inspired by Meta's REA (Ranking Engineer Agent) architecture.

Purpose:
- Store experiment results with metadata
- Identify patterns across successful/failed experiments
- Generate insights for future hypothesis generation
- Enable autonomous learning from past work

Design:
- PostgreSQL storage for structured experiment data
- Embeddings in Qdrant for semantic search
- Automated insight extraction using LLM
- Pattern recognition across experiment dimensions
"""
import os
import json
import psycopg2
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


class HypothesisDatabase:
    """Stores and analyzes experiment hypotheses and results."""

    def __init__(self):
        self.conn = psycopg2.connect(os.environ['POSTGRES_DSN'])
        self._ensure_tables()

    def _ensure_tables(self):
        """Create tables if they don't exist."""
        with self.conn.cursor() as cur:
            # Hypotheses table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS hypotheses (
                    id SERIAL PRIMARY KEY,
                    competition TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    hypothesis TEXT NOT NULL,
                    rationale TEXT,
                    category TEXT,  -- feature_engineering, model_architecture, validation_strategy, etc.
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Results table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS hypothesis_results (
                    id SERIAL PRIMARY KEY,
                    hypothesis_id INTEGER REFERENCES hypotheses(id),
                    cv_score FLOAT,
                    lb_score FLOAT,
                    improvement_vs_baseline FLOAT,
                    succeeded BOOLEAN,
                    failure_reason TEXT,
                    features JSONB,
                    params JSONB,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Insights table (patterns learned from experiments)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS experiment_insights (
                    id SERIAL PRIMARY KEY,
                    competition TEXT NOT NULL,
                    insight TEXT NOT NULL,
                    evidence JSONB,  -- supporting experiments
                    confidence FLOAT,  -- 0-1 score
                    category TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Index for fast lookups
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_hypotheses_competition
                ON hypotheses(competition)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_insights_competition
                ON experiment_insights(competition)
            """)

            self.conn.commit()

    def add_hypothesis(
        self,
        competition: str,
        experiment_id: str,
        hypothesis: str,
        rationale: str,
        category: str
    ) -> int:
        """Add a new hypothesis to the database."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO hypotheses (competition, experiment_id, hypothesis, rationale, category)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (competition, experiment_id, hypothesis, rationale, category))

            hypothesis_id = cur.fetchone()[0]
            self.conn.commit()
            return hypothesis_id

    def record_result(
        self,
        hypothesis_id: int,
        cv_score: float,
        lb_score: Optional[float],
        baseline_cv: float,
        succeeded: bool,
        failure_reason: Optional[str] = None,
        features: Optional[List[str]] = None,
        params: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """Record the result of testing a hypothesis."""
        improvement = (baseline_cv - cv_score) / baseline_cv if cv_score else None

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO hypothesis_results
                (hypothesis_id, cv_score, lb_score, improvement_vs_baseline,
                 succeeded, failure_reason, features, params, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                hypothesis_id, cv_score, lb_score, improvement,
                succeeded, failure_reason,
                json.dumps(features) if features else None,
                json.dumps(params) if params else None,
                json.dumps(metadata) if metadata else None
            ))
            self.conn.commit()

    def get_competition_insights(self, competition: str, min_confidence: float = 0.5) -> List[Dict]:
        """Get insights for a specific competition."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT insight, evidence, confidence, category, created_at
                FROM experiment_insights
                WHERE competition = %s AND confidence >= %s
                ORDER BY confidence DESC, created_at DESC
            """, (competition, min_confidence))

            return [
                {
                    'insight': row[0],
                    'evidence': row[1],
                    'confidence': row[2],
                    'category': row[3],
                    'created_at': row[4]
                }
                for row in cur.fetchall()
            ]

    def analyze_patterns(self, competition: str) -> List[Dict]:
        """Analyze experiment patterns and generate insights."""
        with self.conn.cursor() as cur:
            # Get all hypotheses and results for this competition
            cur.execute("""
                SELECT
                    h.category,
                    h.hypothesis,
                    h.rationale,
                    r.cv_score,
                    r.improvement_vs_baseline,
                    r.succeeded,
                    r.features,
                    r.params
                FROM hypotheses h
                JOIN hypothesis_results r ON h.id = r.hypothesis_id
                WHERE h.competition = %s
                ORDER BY h.created_at DESC
            """, (competition,))

            experiments = cur.fetchall()

        if not experiments:
            return []

        # Analyze by category
        patterns = []
        categories = {}

        for exp in experiments:
            category = exp[0]
            if category not in categories:
                categories[category] = {'succeeded': [], 'failed': []}

            if exp[5]:  # succeeded
                categories[category]['succeeded'].append(exp)
            else:
                categories[category]['failed'].append(exp)

        # Generate insights for each category
        for category, results in categories.items():
            n_succeeded = len(results['succeeded'])
            n_failed = len(results['failed'])
            total = n_succeeded + n_failed

            if total >= 2:  # Need at least 2 experiments to identify pattern
                success_rate = n_succeeded / total

                # High failure rate = insight about what doesn't work
                if success_rate < 0.3 and n_failed >= 2:
                    patterns.append({
                        'category': category,
                        'insight': f"{category} approaches consistently fail",
                        'confidence': min(0.9, n_failed / 5.0),  # Higher confidence with more evidence
                        'evidence': {
                            'failed_count': n_failed,
                            'total_count': total,
                            'examples': [exp[1] for exp in results['failed'][:3]]
                        }
                    })

                # High success rate = insight about what works
                elif success_rate > 0.7 and n_succeeded >= 2:
                    avg_improvement = np.mean([exp[4] for exp in results['succeeded'] if exp[4]])
                    patterns.append({
                        'category': category,
                        'insight': f"{category} approaches show promise (avg {avg_improvement:.1%} improvement)",
                        'confidence': min(0.9, n_succeeded / 5.0),
                        'evidence': {
                            'succeeded_count': n_succeeded,
                            'total_count': total,
                            'avg_improvement': avg_improvement,
                            'examples': [exp[1] for exp in results['succeeded'][:3]]
                        }
                    })

        return patterns

    def store_insight(
        self,
        competition: str,
        insight: str,
        evidence: Dict,
        confidence: float,
        category: str
    ):
        """Store a learned insight."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO experiment_insights (competition, insight, evidence, confidence, category)
                VALUES (%s, %s, %s, %s, %s)
            """, (competition, insight, json.dumps(evidence), confidence, category))
            self.conn.commit()

    def get_experiment_history(
        self,
        competition: str,
        limit: int = 10,
        succeeded_only: bool = False
    ) -> List[Dict]:
        """Get recent experiment history."""
        with self.conn.cursor() as cur:
            query = """
                SELECT
                    h.experiment_id,
                    h.hypothesis,
                    h.category,
                    r.cv_score,
                    r.improvement_vs_baseline,
                    r.succeeded,
                    h.created_at
                FROM hypotheses h
                JOIN hypothesis_results r ON h.id = r.hypothesis_id
                WHERE h.competition = %s
            """

            if succeeded_only:
                query += " AND r.succeeded = TRUE"

            query += " ORDER BY h.created_at DESC LIMIT %s"

            cur.execute(query, (competition, limit))

            return [
                {
                    'experiment_id': row[0],
                    'hypothesis': row[1],
                    'category': row[2],
                    'cv_score': row[3],
                    'improvement': row[4],
                    'succeeded': row[5],
                    'created_at': row[6]
                }
                for row in cur.fetchall()
            ]

    def suggest_next_experiments(self, competition: str, n: int = 3) -> List[Dict]:
        """Generate suggestions for next experiments based on patterns."""
        insights = self.get_competition_insights(competition)
        history = self.get_experiment_history(competition, limit=20)

        suggestions = []

        # Analyze what hasn't been tried yet
        tried_categories = set(exp['category'] for exp in history)
        all_categories = [
            'feature_engineering',
            'model_architecture',
            'validation_strategy',
            'ensemble_methods',
            'data_preprocessing',
            'hyperparameter_tuning'
        ]

        untried = [cat for cat in all_categories if cat not in tried_categories]

        # Suggest untried categories
        for category in untried[:n]:
            suggestions.append({
                'category': category,
                'suggestion': f"Try {category.replace('_', ' ')} approaches",
                'rationale': "Not yet explored in this competition",
                'priority': 'medium'
            })

        # Suggest based on successful patterns
        successful_categories = {}
        for exp in history:
            if exp['succeeded']:
                cat = exp['category']
                if cat not in successful_categories:
                    successful_categories[cat] = []
                successful_categories[cat].append(exp)

        # Iterate on successful approaches
        for category, exps in successful_categories.items():
            if len(suggestions) < n:
                suggestions.append({
                    'category': category,
                    'suggestion': f"Iterate on successful {category.replace('_', ' ')} approach",
                    'rationale': f"{len(exps)} successful experiments in this category",
                    'priority': 'high'
                })

        return suggestions[:n]


def example_usage():
    """Example of how to use the hypothesis database."""
    db = HypothesisDatabase()

    # Record a hypothesis
    hyp_id = db.add_hypothesis(
        competition="store-sales-time-series-forecasting",
        experiment_id="v46",
        hypothesis="Using only recent data (last 1.5 years) improves generalization",
        rationale="Top Kaggle solutions show recent data generalizes better to test set",
        category="data_preprocessing"
    )

    # Record result
    db.record_result(
        hypothesis_id=hyp_id,
        cv_score=0.5445,
        lb_score=None,
        baseline_cv=0.3572,
        succeeded=False,
        failure_reason="Less training data hurt performance despite better temporal alignment",
        features=['Lag_3', 'Lag_7', 'Roll_mean_7', 'Roll_mean_14'],
        params={'learning_rate': 0.05, 'max_depth': 6}
    )

    # Analyze patterns
    patterns = db.analyze_patterns("store-sales-time-series-forecasting")
    for pattern in patterns:
        print(f"Pattern: {pattern['insight']} (confidence: {pattern['confidence']:.2f})")

        # Store as insight
        db.store_insight(
            competition="store-sales-time-series-forecasting",
            insight=pattern['insight'],
            evidence=pattern['evidence'],
            confidence=pattern['confidence'],
            category=pattern['category']
        )

    # Get suggestions
    suggestions = db.suggest_next_experiments("store-sales-time-series-forecasting")
    for sugg in suggestions:
        print(f"Suggestion: {sugg['suggestion']} ({sugg['priority']} priority)")


if __name__ == "__main__":
    example_usage()
