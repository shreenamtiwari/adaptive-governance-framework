# ============================================================================
# Adaptive Data Governance Framework
# src/pii_detection/adaptive_pii_tuner.py
# ============================================================================
# Feedback-Driven PII Confidence Tuning
#
# The PII detector uses a static confidence_threshold (default 0.85).
# This module adds a feedback loop that:
#
#   1. Logs every PII detection event (entity type, score, user feedback).
#   2. Computes precision/recall per entity type from accumulated feedback.
#   3. Auto-adjusts the confidence threshold per entity type to maximise
#      the F1 score.
#   4. Detects PII-type drift (e.g. a new address format appearing that
#      the regex doesn't catch) by monitoring false-negative rates.
#
# The tuner persists its state as a JSON file so adjustments survive
# restarts and improve over successive pipeline runs.
# ============================================================================

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


# ============================================================================
# Feedback event schema
# ============================================================================

class PIIFeedbackEvent:
    """A single feedback observation about a PII detection."""

    __slots__ = (
        "entity_type", "text", "score", "predicted_pii",
        "actual_pii", "timestamp",
    )

    def __init__(
        self,
        entity_type: str,
        text: str,
        score: float,
        predicted_pii: bool,
        actual_pii: bool,
        timestamp: Optional[str] = None,
    ):
        self.entity_type = entity_type
        self.text = text
        self.score = score
        self.predicted_pii = predicted_pii
        self.actual_pii = actual_pii
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "text": self.text,
            "score": self.score,
            "predicted_pii": self.predicted_pii,
            "actual_pii": self.actual_pii,
            "timestamp": self.timestamp,
        }


# ============================================================================
# AdaptivePIITuner
# ============================================================================


class AdaptivePIITuner:
    """Feedback-driven confidence threshold tuner for PII detection.

    Parameters
    ----------
    feedback_dir : str
        Directory where feedback events and tuned thresholds are stored.
    default_threshold : float
        Starting confidence threshold before any tuning (default 0.85).
    min_threshold : float
        Minimum confidence threshold (floor) to prevent over-sensitivity.
    max_threshold : float
        Maximum confidence threshold (ceiling) to prevent under-detection.
    min_feedback_count : int
        Minimum number of feedback events per entity type before
        threshold adjustment begins.
    """

    def __init__(
        self,
        feedback_dir: str = "data/metrics/pii_feedback",
        default_threshold: float = 0.85,
        min_threshold: float = 0.50,
        max_threshold: float = 0.99,
        min_feedback_count: int = 20,
    ):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.default_threshold = default_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.min_feedback_count = min_feedback_count

    # ------------------------------------------------------------------
    # Record feedback
    # ------------------------------------------------------------------

    def record_feedback(self, event: PIIFeedbackEvent) -> None:
        """Append a feedback event to the persistent log."""
        feedback = self._load_feedback()
        feedback.append(event.to_dict())
        self._save_feedback(feedback)
        logger.debug(
            "PII feedback recorded: type={}, predicted={}, actual={}",
            event.entity_type, event.predicted_pii, event.actual_pii,
        )

    def record_batch_feedback(
        self,
        events: List[PIIFeedbackEvent],
    ) -> None:
        """Append multiple feedback events at once."""
        feedback = self._load_feedback()
        for e in events:
            feedback.append(e.to_dict())
        self._save_feedback(feedback)
        logger.info("PII feedback: {} events recorded", len(events))

    # ------------------------------------------------------------------
    # Compute per-entity-type metrics
    # ------------------------------------------------------------------

    def compute_entity_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Compute precision, recall, and F1 per entity type.

        Returns
        -------
        dict[str, dict]
            ``{entity_type: {tp, fp, fn, tn, precision, recall, f1, count}}``
        """
        feedback = self._load_feedback()
        if not feedback:
            return {}

        # Group by entity type
        by_type: Dict[str, List[Dict]] = {}
        for event in feedback:
            et = event["entity_type"]
            by_type.setdefault(et, []).append(event)

        metrics: Dict[str, Dict[str, Any]] = {}
        for et, events in by_type.items():
            tp = sum(1 for e in events if e["predicted_pii"] and e["actual_pii"])
            fp = sum(1 for e in events if e["predicted_pii"] and not e["actual_pii"])
            fn = sum(1 for e in events if not e["predicted_pii"] and e["actual_pii"])
            tn = sum(1 for e in events if not e["predicted_pii"] and not e["actual_pii"])

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            metrics[et] = {
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "count": len(events),
            }

        return metrics

    # ------------------------------------------------------------------
    # Auto-tune thresholds
    # ------------------------------------------------------------------

    def tune_thresholds(self) -> Dict[str, float]:
        """Auto-adjust confidence thresholds per entity type.

        Strategy: For each entity type with enough feedback, find the
        threshold that maximises the F1 score by testing a grid of
        candidate thresholds from the observed score distribution.

        Returns
        -------
        dict[str, float]
            ``{entity_type: optimal_threshold}``
        """
        feedback = self._load_feedback()
        if not feedback:
            logger.info("No PII feedback available — using default thresholds")
            return {}

        by_type: Dict[str, List[Dict]] = {}
        for event in feedback:
            by_type.setdefault(event["entity_type"], []).append(event)

        tuned: Dict[str, float] = {}

        for et, events in by_type.items():
            if len(events) < self.min_feedback_count:
                logger.debug(
                    "Entity '{}': only {} events (need {}) — skip tuning",
                    et, len(events), self.min_feedback_count,
                )
                tuned[et] = self.default_threshold
                continue

            # Extract scores and ground truth
            scores = np.array([e["score"] for e in events])
            actual = np.array([e["actual_pii"] for e in events])

            # Test candidate thresholds
            candidates = np.linspace(
                self.min_threshold, self.max_threshold, 50
            )

            best_f1 = -1.0
            best_thresh = self.default_threshold

            for t in candidates:
                predicted = scores >= t
                tp = np.sum(predicted & actual)
                fp = np.sum(predicted & ~actual)
                fn = np.sum(~predicted & actual)

                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = float(t)

            tuned[et] = round(best_thresh, 4)

            logger.info(
                "Entity '{}': optimal threshold={:.4f} (F1={:.4f}, n={})",
                et, best_thresh, best_f1, len(events),
            )

        # Persist tuned thresholds
        self._save_tuned_thresholds(tuned)
        return tuned

    # ------------------------------------------------------------------
    # Get thresholds for use by PIIDetector
    # ------------------------------------------------------------------

    def get_thresholds(self) -> Dict[str, float]:
        """Load the most recently tuned thresholds.

        If no tuned thresholds exist, returns an empty dict (detector
        should fall back to its own default).
        """
        path = self.feedback_dir / "tuned_thresholds.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                # Handle both flat dict and nested {"thresholds": {...}} format
                return data.get("thresholds", data) if isinstance(data, dict) else {}
        return {}

    def get_optimal_threshold(self, entity_type: str) -> float:
        """Return the optimal confidence threshold for a specific entity type.

        Falls back to ``self.default_threshold`` if no tuned threshold
        exists for the requested entity type.

        Parameters
        ----------
        entity_type : str
            PII entity type (e.g. ``"PERSON"``, ``"EMAIL"``).

        Returns
        -------
        float
        """
        thresholds = self.get_thresholds()
        return thresholds.get(entity_type, self.default_threshold)

    def should_use_conservative_mode(self) -> bool:
        """Check if conservative masking mode should be activated.

        Conservative mode lowers all thresholds by 20 %% when PII drift
        is detected (FN rate delta > 5 %%).  This prevents data leaks
        until the model is re-tuned.

        Returns
        -------
        bool
        """
        drift_report = self.detect_pii_drift()
        return drift_report.get("has_drift", False)

    # ------------------------------------------------------------------
    # PII-type drift detection
    # ------------------------------------------------------------------

    def detect_pii_drift(self) -> Dict[str, Any]:
        """Detect shifts in PII detection patterns over time.

        Compares recent feedback (last 25 %) against historical baseline
        (first 75 %) to find:
        - Entity types with increasing false-negative rates (undetected PII)
        - New entity types not seen in the baseline period
        - Accuracy degradation per type

        Returns
        -------
        dict
            Drift report with ``has_drift``, ``drifted_types``, etc.
        """
        feedback = self._load_feedback()
        if len(feedback) < 20:
            return {"has_drift": False, "reason": "insufficient_data"}

        split = int(len(feedback) * 0.75)
        baseline = feedback[:split]
        recent = feedback[split:]

        def _fn_rate(events: List[Dict]) -> Dict[str, float]:
            by_type: Dict[str, List[Dict]] = {}
            for e in events:
                by_type.setdefault(e["entity_type"], []).append(e)
            rates = {}
            for et, evts in by_type.items():
                fn = sum(1 for e in evts if not e["predicted_pii"] and e["actual_pii"])
                total = len(evts)
                rates[et] = fn / total if total else 0.0
            return rates

        baseline_rates = _fn_rate(baseline)
        recent_rates = _fn_rate(recent)

        drifted = []
        for et, recent_fn in recent_rates.items():
            baseline_fn = baseline_rates.get(et, 0.0)
            if recent_fn - baseline_fn > 0.05:  # >5 pp increase in FN rate
                drifted.append({
                    "entity_type": et,
                    "baseline_fn_rate": round(baseline_fn, 4),
                    "recent_fn_rate": round(recent_fn, 4),
                    "delta": round(recent_fn - baseline_fn, 4),
                })

        # New entity types
        baseline_types = set(e["entity_type"] for e in baseline)
        recent_types = set(e["entity_type"] for e in recent)
        new_types = list(recent_types - baseline_types)

        report = {
            "has_drift": bool(drifted or new_types),
            "drifted_types": drifted,
            "new_entity_types": new_types,
            "baseline_size": len(baseline),
            "recent_size": len(recent),
            "timestamp": datetime.now().isoformat(),
        }

        if report["has_drift"]:
            logger.warning(
                "PII drift detected: {} drifted types, {} new types",
                len(drifted), len(new_types),
            )

        return report

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _feedback_file(self) -> Path:
        return self.feedback_dir / "pii_feedback_log.json"

    def _load_feedback(self) -> List[Dict]:
        path = self._feedback_file()
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return []

    def _save_feedback(self, feedback: List[Dict]) -> None:
        # Keep at most 10 000 events
        feedback = feedback[-10_000:]
        with open(self._feedback_file(), "w") as f:
            json.dump(feedback, f, indent=2)

    def _save_tuned_thresholds(self, thresholds: Dict[str, float]) -> None:
        path = self.feedback_dir / "tuned_thresholds.json"
        payload = {
            "thresholds": thresholds,
            "tuned_at": datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Tuned thresholds saved → {}", path)
