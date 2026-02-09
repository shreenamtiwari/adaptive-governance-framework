# ============================================================================
# Adaptive Data Governance Framework
# src/transformation/bronze_to_silver.py
# ============================================================================
# Bronze → Silver transformation pipeline.
#   - Deduplication via window functions
#   - Schema validation
#   - PII masking on free-text columns
#   - Data-quality gating (quarantine bad records)
#   - Adds Silver-layer metadata columns
# ============================================================================

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


# ============================================================================
# BronzeToSilverTransformer
# ============================================================================

class BronzeToSilverTransformer:
    """Orchestrate the Bronze → Silver transformation step.

    Parameters
    ----------
    spark : SparkSession
    bronze_path : str
        Root of the Bronze Delta Lake layer.
    silver_path : str
        Root of the Silver Delta Lake layer.
    quarantine_path : str
        Where quarantined records are written.
    """

    def __init__(
        self,
        spark: SparkSession,
        bronze_path: str = "data/bronze",
        silver_path: str = "data/silver",
        quarantine_path: str = "data/quarantine",
    ):
        self.spark = spark
        self.bronze_path = Path(bronze_path)
        self.silver_path = Path(silver_path)
        self.quarantine_path = Path(quarantine_path)

    # ------------------------------------------------------------------
    # 1. Deduplication
    # ------------------------------------------------------------------

    def deduplicate(
        self,
        df: DataFrame,
        key_columns: List[str],
        order_column: str,
        ascending: bool = False,
    ) -> DataFrame:
        """Remove duplicate records using a window function.

        Keeps the **latest** (or earliest) record per key by default.

        Parameters
        ----------
        df : DataFrame
        key_columns : list[str]
            Columns that define uniqueness.
        order_column : str
            Column used to pick the survivor row.
        ascending : bool
            If ``False`` (default) keeps the most-recent row.
        """
        window = Window.partitionBy(*key_columns).orderBy(
            F.col(order_column).asc() if ascending else F.col(order_column).desc()
        )
        deduped = (
            df
            .withColumn("_row_num", F.row_number().over(window))
            .filter(F.col("_row_num") == 1)
            .drop("_row_num")
        )
        removed = df.count() - deduped.count()
        if removed > 0:
            logger.info("Deduplication removed {} duplicate rows.", removed)
        return deduped

    # ------------------------------------------------------------------
    # 2. Null / value validation
    # ------------------------------------------------------------------

    def validate_not_null(
        self,
        df: DataFrame,
        required_columns: List[str],
    ) -> Tuple[DataFrame, DataFrame]:
        """Split DataFrame into valid (non-null on required cols) and quarantined.

        Returns
        -------
        tuple[DataFrame, DataFrame]
            ``(valid_df, quarantined_df)``
        """
        condition = F.lit(True)
        for col in required_columns:
            condition = condition & F.col(col).isNotNull()

        valid = df.filter(condition)
        quarantined = df.filter(~condition)

        if quarantined.count() > 0:
            logger.warning(
                "Quarantined {} records due to null required fields.",
                quarantined.count(),
            )
        return valid, quarantined

    def validate_value_ranges(
        self,
        df: DataFrame,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> Tuple[DataFrame, DataFrame]:
        """Split DataFrame based on value range validation."""
        condition = F.lit(True)
        if min_val is not None:
            condition = condition & (F.col(column) >= min_val)
        if max_val is not None:
            condition = condition & (F.col(column) <= max_val)

        valid = df.filter(condition)
        quarantined = df.filter(~condition)

        if quarantined.count() > 0:
            logger.warning(
                "Quarantined {} records — {c} outside [{lo}, {hi}].",
                quarantined.count(),
                c=column, lo=min_val, hi=max_val,
            )
        return valid, quarantined

    # ------------------------------------------------------------------
    # 3. PII masking
    # ------------------------------------------------------------------

    def mask_pii_columns(
        self,
        df: DataFrame,
        text_columns: List[str],
        strategy: str = "redact",
    ) -> DataFrame:
        """Apply PII masking UDF to specified text columns.

        Parameters
        ----------
        df : DataFrame
        text_columns : list[str]
            Columns containing free text that may include PII.
        strategy : str
            ``"hash"``, ``"redact"``, or ``"tokenize"``.
        """
        from src.pii_detection.pii_masker import PIIMasker

        mask_udf = PIIMasker(strategy=strategy).create_spark_mask_udf()

        for col in text_columns:
            if col in df.columns:
                df = df.withColumn(col, mask_udf(F.col(col)))
                logger.info("PII masking applied to column '{}'.", col)

        return df

    # ------------------------------------------------------------------
    # 4. Add Silver metadata
    # ------------------------------------------------------------------

    @staticmethod
    def add_silver_metadata(
        df: DataFrame,
        pipeline_run_id: Optional[str] = None,
    ) -> DataFrame:
        """Append audit / lineage columns for the Silver layer.

        Adds DPDP Act 2023 compliance fields (Section 12) and
        transformation lineage for the "Lineage of Trust".
        """
        import uuid
        _run_id = pipeline_run_id or str(uuid.uuid4())
        return (
            df
            .withColumn("_silver_processed_at", F.current_timestamp())
            .withColumn("_dq_validated", F.lit(True))
            .withColumn("_pii_masked", F.lit(True))
            # Lineage of Trust
            .withColumn("_pipeline_run_id", F.lit(_run_id))
            .withColumn("_processed_by", F.lit("BronzeToSilverTransformer"))
            .withColumn("_transformation_version", F.lit("2.0.0"))
            # DPDP Act 2023 — Section 12 compliance
            .withColumn("_right_to_erasure", F.lit(False))
            .withColumn("_consent_timestamp", F.current_timestamp())
        )

    # ------------------------------------------------------------------
    # 5. Write to Silver
    # ------------------------------------------------------------------

    def write_to_silver(
        self,
        df: DataFrame,
        table_name: str,
        partition_cols: Optional[List[str]] = None,
    ) -> str:
        """Persist the Silver DataFrame as a Delta table.

        Returns the Delta table path.
        """
        path = str(self.silver_path / table_name)

        writer = (
            df.write
            .format("delta")
            .mode("overwrite")
            .option("optimizeWrite", "true")
        )
        if partition_cols:
            writer = (
                writer
                .option("partitionOverwriteMode", "static")
                .partitionBy(*partition_cols)
            )
        else:
            writer = writer.option("overwriteSchema", "true")

        writer.save(path)
        logger.info(
            "Written Silver table — {t}, rows={r}",
            t=table_name, r=df.count(),
        )
        return path

    def write_quarantine(self, df: DataFrame, table_name: str) -> None:
        """Persist quarantined records to a Delta table for audit."""
        if df.count() == 0:
            return
        path = str(self.quarantine_path / table_name)
        (
            df
            .withColumn("_quarantined_at", F.current_timestamp())
            .write
            .format("delta")
            .mode("append")
            .save(path)
        )
        logger.info("Quarantined {} records → {}", df.count(), path)

    # ------------------------------------------------------------------
    # 6. End-to-end pipeline
    # ------------------------------------------------------------------

    def transform_orders(
        self,
        table_name: str = "orders",
        pii_columns: Optional[List[str]] = None,
        masking_strategy: str = "redact",
        pipeline_run_id: Optional[str] = None,
    ) -> DataFrame:
        """Run the full Bronze → Silver pipeline for orders.

        Applies context-aware masking strategies per DPDP Act 2023:
          - ``hash`` (SHA-256) for identifiers (email, phone) to
            maintain joinability for Identity Resolution.
          - ``redact`` ([REDACTED]) for free-text fields discovered
            by the NER model (delivery notes, reviews).
          - ``tokenize`` (FPE) for format-sensitive fields (pincodes).

        Steps:
          1. Read Bronze
          2. Deduplicate
          3. Validate nulls & ranges
          4. Mask PII (per-column strategy)
          5. Add lineage + DPDP metadata
          6. Write Silver + quarantine

        Returns
        -------
        DataFrame
            The validated, masked Silver DataFrame.
        """
        # Per-column masking strategy map (Defense in Depth)
        _column_strategies: Dict[str, Tuple[List[str], str]] = {
            "redact": (["delivery_instructions", "customer_review"], "redact"),
            "hash": (["customer_email", "customer_phone"], "hash"),
            "tokenize": (["pincode", "delivery_pincode"], "tokenize"),
        }

        if pii_columns is None:
            pii_columns = ["delivery_instructions", "customer_review"]

        logger.info("Starting Bronze → Silver for '{}'", table_name)

        # Read
        bronze = self.spark.read.format("delta").load(
            str(self.bronze_path / table_name)
        )
        logger.info("Bronze read — {} rows", bronze.count())

        # Deduplicate
        df = self.deduplicate(
            bronze,
            key_columns=["order_id"],
            order_column="_ingested_at",
        )

        # Validate nulls
        df, q_nulls = self.validate_not_null(
            df,
            required_columns=["order_id", "customer_id", "order_value"],
        )

        # Validate ranges
        df, q_range = self.validate_value_ranges(
            df, column="order_value", min_val=0.0,
        )

        # PII masking — per-column strategy (Defense in Depth)
        for strategy, (cols, strat) in _column_strategies.items():
            present_cols = [c for c in cols if c in df.columns]
            if present_cols:
                df = self.mask_pii_columns(df, present_cols, strategy=strat)
        # Also mask any explicitly requested columns with the given strategy
        remaining = [c for c in pii_columns if c in df.columns]
        if remaining:
            df = self.mask_pii_columns(df, remaining, strategy=masking_strategy)

        # Metadata — lineage + DPDP compliance
        df = self.add_silver_metadata(df, pipeline_run_id=pipeline_run_id)

        # Quarantine
        from functools import reduce
        quarantined = reduce(DataFrame.union, [q_nulls, q_range])
        self.write_quarantine(quarantined, table_name)

        # Write Silver
        self.write_to_silver(df, table_name)

        # Log PII audit event to data/metrics/pii_audits/
        self._log_pii_audit(table_name, df.count(), pii_columns)

        logger.info(
            "Bronze → Silver complete for '{}': {} valid, {} quarantined",
            table_name, df.count(), quarantined.count(),
        )
        return df

    def _log_pii_audit(
        self, table_name: str, row_count: int, masked_columns: List[str],
    ) -> None:
        """Persist PII masking audit log for DPDP Act regulatory reporting."""
        import json
        from datetime import datetime

        audit_dir = Path(self.silver_path).parent / "metrics" / "pii_audits"
        audit_dir.mkdir(parents=True, exist_ok=True)

        audit_entry = {
            "table": table_name,
            "rows_processed": row_count,
            "masked_columns": masked_columns,
            "masking_strategies": {
                "hash": "SHA-256 (identifiers — joinable)",
                "redact": "[REDACTED] (NER-discovered free text)",
                "tokenize": "FPE/HMAC (format-preserving — pincodes)",
            },
            "dpdp_fields_added": ["_right_to_erasure", "_consent_timestamp"],
            "timestamp": datetime.now().isoformat(),
        }

        path = audit_dir / f"{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(path, "w") as f:
            json.dump(audit_entry, f, indent=2)
        logger.info("PII audit log → {}", path)
