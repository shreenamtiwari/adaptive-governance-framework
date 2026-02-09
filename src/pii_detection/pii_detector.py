# ============================================================================
# Adaptive Data Governance Framework
# src/pii_detection/pii_detector.py
# ============================================================================
# Transformer-based + regex-based PII detection engine.
# Uses Presidio for structured PII (email, phone, Aadhaar, PAN, credit card)
# and Hugging Face DistilBERT NER for unstructured person-name detection.
# Registers a Spark UDF for distributed processing.
# ============================================================================

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# PII Entity dataclass
# ---------------------------------------------------------------------------

@dataclass
class PIIEntity:
    """A single detected PII span."""
    entity_type: str
    text: str
    start: int
    end: int
    score: float  # 0.0 – 1.0 confidence


# ---------------------------------------------------------------------------
# Regex patterns for Indian PII
# ---------------------------------------------------------------------------

_REGEX_PATTERNS: Dict[str, re.Pattern] = {
    "EMAIL": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),
    "PHONE_NUMBER": re.compile(
        r"(\+91[\-\s]?\d{10})"
        r"|(\b0\d{2,4}[\-\s]?\d{6,8}\b)"
        r"|(\b[6-9]\d{9}\b)"
    ),
    "AADHAAR": re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),
    "PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "CREDIT_CARD": re.compile(
        r"\b(?:\d[ \-]*?){13,19}\b"
    ),
    "IPV4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "ADDRESS": re.compile(
        # Indian address patterns: "H.No 12, Sector 5, Noida, UP 201301"
        r"\b(?:H\.?\s*No\.?|Flat|Plot|House|Apt)\.?\s*\d+"
        r"[\w\s,\-\.]*\d{6}\b",
        re.IGNORECASE,
    ),
    "IFSC": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
}


# ============================================================================
# PIIDetector
# ============================================================================

class PIIDetector:
    """Detect PII in free-text using regex patterns and (optionally) a
    Hugging Face NER model for person-name detection.

    Parameters
    ----------
    confidence_threshold : float
        Minimum confidence to report a PII entity.
    use_ner_model : bool
        Whether to load a transformer NER model for PERSON detection.
    model_name : str
        Hugging Face model identifier.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        use_ner_model: bool = False,
        model_name: str = "dslim/bert-base-NER",
        adaptive_thresholds: Optional[Dict[str, float]] = None,
        conservative_mode: bool = False,
    ):
        self.confidence_threshold = confidence_threshold
        self.use_ner_model = use_ner_model
        self._ner_pipeline = None
        # Per-entity-type thresholds from AdaptivePIITuner
        self._adaptive_thresholds = adaptive_thresholds or {}
        # Conservative mode: lower all thresholds by 20% when PII drift detected
        self.conservative_mode = conservative_mode
        if conservative_mode:
            self.confidence_threshold = max(0.50, confidence_threshold * 0.80)
            self._adaptive_thresholds = {
                et: max(0.50, t * 0.80)
                for et, t in self._adaptive_thresholds.items()
            }

        if use_ner_model:
            self._load_ner_model(model_name)

    def _load_ner_model(self, model_name: str) -> None:
        """Lazy-load the Hugging Face NER pipeline."""
        try:
            from transformers import pipeline as hf_pipeline
            self._ner_pipeline = hf_pipeline(
                "ner",
                model=model_name,
                aggregation_strategy="simple",
            )
            logger.info("NER model loaded: {}", model_name)
        except Exception as exc:
            logger.warning("Could not load NER model: {}", exc)
            self._ner_pipeline = None

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_pii(self, text: str) -> List[PIIEntity]:
        """Scan *text* and return all detected PII entities.

        Parameters
        ----------
        text : str
            Input text to scan.

        Returns
        -------
        list[PIIEntity]
        """
        if not text or not isinstance(text, str):
            return []

        entities: List[PIIEntity] = []

        # 1. Regex-based detection
        for entity_type, pattern in _REGEX_PATTERNS.items():
            for match in pattern.finditer(text):
                entities.append(
                    PIIEntity(
                        entity_type=entity_type,
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        score=1.0,  # Regex matches are deterministic
                    )
                )

        # 2. NER-based person-name detection (chunked for long texts)
        if self._ner_pipeline is not None:
            try:
                # Process in 450-char chunks with 50-char overlap to
                # avoid splitting entities at boundaries.
                chunk_size = 450
                overlap = 50
                text_len = len(text)
                chunks = []
                start = 0
                while start < text_len:
                    end = min(start + chunk_size, text_len)
                    chunks.append((start, text[start:end]))
                    start += chunk_size - overlap

                seen_spans = set()
                for chunk_offset, chunk_text in chunks:
                    ner_results = self._ner_pipeline(chunk_text)
                for ent in ner_results:
                    # Use adaptive per-entity threshold if available
                    _ent_thresh = self._adaptive_thresholds.get(
                        "PERSON", self.confidence_threshold,
                    )
                    if (
                            ent["entity_group"] == "PER"
                            and ent["score"] >= _ent_thresh
                        ):
                            abs_start = chunk_offset + ent["start"]
                            abs_end = chunk_offset + ent["end"]
                            span_key = (abs_start, abs_end)
                            if span_key not in seen_spans:
                                seen_spans.add(span_key)
                                entities.append(
                                    PIIEntity(
                                        entity_type="PERSON",
                                        text=ent["word"],
                                        start=abs_start,
                                        end=abs_end,
                                        score=round(ent["score"], 4),
                                    )
                                )
            except Exception as exc:
                logger.error("NER inference error: {}", exc)

        # Filter by per-entity adaptive threshold (falls back to default)
        entities = [
            e for e in entities
            if e.score >= self._adaptive_thresholds.get(
                e.entity_type, self.confidence_threshold,
            )
        ]

        return entities

    def detect_pii_types(self, text: str) -> Dict[str, int]:
        """Return a dict of {entity_type: count} for the given text."""
        entities = self.detect_pii(text)
        counts: Dict[str, int] = {}
        for e in entities:
            counts[e.entity_type] = counts.get(e.entity_type, 0) + 1
        return counts

    def has_pii(self, text: str) -> bool:
        """Quick check: does the text contain any PII?"""
        return len(self.detect_pii(text)) > 0

    # ------------------------------------------------------------------
    # Spark UDF factory
    # ------------------------------------------------------------------

    @staticmethod
    def create_spark_detect_udf(use_ner: bool = False):
        """Return a PySpark UDF that detects PII and returns a JSON string
        of entities.

        Parameters
        ----------
        use_ner : bool
            Whether to load the NER model inside the UDF.
            Requires the model to be accessible on each executor.

        Usage::

            from pyspark.sql import functions as F
            detect_udf = PIIDetector.create_spark_detect_udf(use_ner=True)
            df = df.withColumn("pii_entities", detect_udf(F.col("text_col")))
        """
        import json
        from pyspark.sql.functions import udf
        from pyspark.sql.types import StringType

        _use_ner = use_ner  # capture in closure

        # Load adaptive thresholds if available
        _adaptive_thresh: Dict[str, float] = {}
        try:
            from src.pii_detection.adaptive_pii_tuner import AdaptivePIITuner
            _adaptive_thresh = AdaptivePIITuner().get_thresholds()
            if isinstance(_adaptive_thresh, dict) and "thresholds" in _adaptive_thresh:
                _adaptive_thresh = _adaptive_thresh["thresholds"]
        except Exception:
            pass

        @udf(returnType=StringType())
        def _detect_pii_udf(text: str) -> str:
            if not text:
                return "[]"
            # Preprocess chat/ticket text — strip role prefixes
            cleaned = _preprocess_chat_text(text)
            detector = PIIDetector(
                confidence_threshold=0.85,
                use_ner_model=_use_ner,
                adaptive_thresholds=_adaptive_thresh,
            )
            entities = detector.detect_pii(cleaned)
            return json.dumps(
                [
                    {
                        "type": e.entity_type,
                        "text": e.text,
                        "start": e.start,
                        "end": e.end,
                        "score": e.score,
                    }
                    for e in entities
                ]
            )

        return _detect_pii_udf


# ============================================================================
# Chat / support-ticket preprocessing
# ============================================================================

def _preprocess_chat_text(text: str) -> str:
    """Strip common chat/ticket role prefixes and system noise.

    Handles patterns like:
        "Agent: How can I help?\nCustomer: My Aadhaar is 1234 5678 9012"
    """
    import re as _re
    # Remove role prefixes (Agent:, Customer:, Bot:, System:, etc.)
    cleaned = _re.sub(
        r"^(Agent|Customer|Bot|System|Support|User)\s*:\s*",
        "",
        text,
        flags=_re.MULTILINE | _re.IGNORECASE,
    )
    # Collapse whitespace
    cleaned = _re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ============================================================================
# Convenience CLI
# ============================================================================

if __name__ == "__main__":
    detector = PIIDetector(confidence_threshold=0.85, use_ner_model=False)

    sample_texts = [
        "Deliver to Rajesh Kumar at rajesh.kumar@gmail.com, phone +91-9876543210",
        "Aadhaar: 1234 5678 9012, PAN: ABCDE1234F",
        "No PII here, just a regular sentence about data governance.",
        "Call me on 9876543210 or email test@example.com",
    ]

    for text in sample_texts:
        entities = detector.detect_pii(text)
        print(f"\nText: {text}")
        for e in entities:
            print(f"  → {e.entity_type}: '{e.text}' (score={e.score})")
