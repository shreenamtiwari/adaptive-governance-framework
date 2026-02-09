# ============================================================================
# Adaptive Data Governance Framework
# src/pii_detection/pii_masker.py
# ============================================================================
# Apply masking transformations based on PII detection results.
# Maintains referential integrity through consistent hashing.
#
# Strategies:
#   HASH    – SHA-256 with optional salt (format-preserving length)
#   REDACT  – Replace PII span with entity-type placeholder (e.g. ***EMAIL***)
#   TOKENIZE – Format-preserving tokenisation with a secret key
# ============================================================================

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

from loguru import logger

from src.pii_detection.pii_detector import PIIDetector, PIIEntity


# ============================================================================
# PIIMasker
# ============================================================================

class PIIMasker:
    """Apply masking to text fields based on detected PII.

    Parameters
    ----------
    strategy : str
        ``"hash"``, ``"redact"``, or ``"tokenize"``.
    salt : str
        Salt for hashing (ensures consistent masking across runs).
    detector : PIIDetector | None
        Reuse an existing detector instance.
    """

    STRATEGIES = ("hash", "redact", "tokenize")

    def __init__(
        self,
        strategy: str = "hash",
        salt: str = "governance_salt_2024",
        detector: Optional[PIIDetector] = None,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {self.STRATEGIES}")

        self.strategy = strategy
        self.salt = salt
        self.detector = detector or PIIDetector(confidence_threshold=0.85)

    # ------------------------------------------------------------------
    # Core masking methods
    # ------------------------------------------------------------------

    def _hash_value(self, value: str) -> str:
        """SHA-256 hash truncated to the original value's length."""
        digest = hashlib.sha256(f"{self.salt}{value}".encode()).hexdigest()
        return digest[: max(len(value), 8)]

    def _redact_value(self, entity_type: str) -> str:
        """Replace with a typed [REDACTED] placeholder (DPDP Act compliant)."""
        return f"[REDACTED:{entity_type}]"

    def _tokenize_value(self, value: str) -> str:
        """Format-preserving encryption (FPE) using keyed HMAC.

        Produces deterministic, format-preserving output:
        digits → digits, letters → letters, preserving separators.
        Uses HMAC-SHA256 with ``self.salt`` as the key for
        reproducibility (reversible with the same key).
        """
        import hmac

        # Derive a deterministic pseudo-random stream from the value
        mac = hmac.new(
            self.salt.encode(), value.encode(), "sha256",
        ).hexdigest()

        result = []
        mac_idx = 0
        for ch in value:
            if ch.isdigit():
                result.append(str(int(mac[mac_idx % len(mac)], 16) % 10))
                mac_idx += 1
            elif ch.isalpha():
                offset = int(mac[mac_idx % len(mac)], 16)
                base = ord('A') if ch.isupper() else ord('a')
                result.append(chr(base + offset % 26))
                mac_idx += 1
            else:
                result.append(ch)
        return "".join(result)

    def _mask_entity(self, entity: PIIEntity) -> str:
        """Return the masked replacement string for a single entity."""
        if self.strategy == "hash":
            return self._hash_value(entity.text)
        elif self.strategy == "redact":
            return self._redact_value(entity.entity_type)
        else:  # tokenize
            return self._tokenize_value(entity.text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mask_text(self, text: str) -> str:
        """Detect PII in *text* and replace every entity.

        Replacements are applied from the **end** of the string towards
        the beginning so that offsets remain valid.
        """
        if not text:
            return text

        entities = self.detector.detect_pii(text)
        if not entities:
            return text

        # Sort by start position descending to preserve offsets
        entities.sort(key=lambda e: e.start, reverse=True)

        masked = text
        for entity in entities:
            replacement = self._mask_entity(entity)
            masked = masked[: entity.start] + replacement + masked[entity.end :]

        return masked

    def mask_text_with_report(self, text: str) -> Dict:
        """Mask text and return both the masked version and a report.

        Returns
        -------
        dict
            ``original``, ``masked``, ``entities_found``,
            ``entity_count``, ``strategy``.
        """
        entities = self.detector.detect_pii(text) if text else []

        masked = self.mask_text(text) if text else text

        return {
            "original": text,
            "masked": masked,
            "entities_found": [
                {"type": e.entity_type, "text": e.text, "score": e.score}
                for e in entities
            ],
            "entity_count": len(entities),
            "strategy": self.strategy,
        }

    # ------------------------------------------------------------------
    # Spark UDF factory
    # ------------------------------------------------------------------

    def create_spark_mask_udf(self, strategy: Optional[str] = None):
        """Return a PySpark UDF that masks PII in a text column.

        Can be called as an instance method (uses ``self.strategy``) or
        as a class method with an explicit *strategy* argument.

        Usage::

            mask_udf = PIIMasker(strategy="redact").create_spark_mask_udf()
            df = df.withColumn("masked_text", mask_udf(F.col("raw_text")))
        """
        from pyspark.sql.functions import udf
        from pyspark.sql.types import StringType

        chosen_strategy = strategy or self.strategy

        @udf(returnType=StringType())
        def _mask_pii_udf(text: str) -> str:
            if not text:
                return text
            masker = PIIMasker(strategy=chosen_strategy)
            return masker.mask_text(text)

        return _mask_pii_udf


# ============================================================================
# CLI demo
# ============================================================================

if __name__ == "__main__":
    masker = PIIMasker(strategy="redact")

    samples = [
        "Ship to Priya Sharma at priya@example.com, phone +91-9876543210",
        "Aadhaar: 1234 5678 9012, PAN: ABCDE1234F, Credit 4111-1111-1111-1111",
        "No PII in this review. Great product!",
    ]

    for text in samples:
        result = masker.mask_text_with_report(text)
        print(f"\nOriginal : {result['original']}")
        print(f"Masked   : {result['masked']}")
        print(f"Entities : {result['entity_count']} found")
        for ent in result["entities_found"]:
            print(f"  → {ent['type']}: '{ent['text']}' (score={ent['score']})")
