# ============================================================================
# Adaptive Data Governance Framework
# src/quality/data_profiler.py
# ============================================================================
# DataProfiler – comprehensive profiling engine for PySpark DataFrames.
#
# Responsibilities:
#   1. Profile every column in a DataFrame: nulls, distinct counts, data-type
#      distribution, numeric statistics, categorical summaries, and PII scans.
#   2. Detect schema drift against a saved JSON reference.
#   3. Persist profiles as JSON snapshots and generate matplotlib/seaborn
#      charts in docs/profiles/.
#
# Usage:
#   >>> from pyspark.sql import SparkSession
#   >>> from src.quality.data_profiler import DataProfiler
#   >>> spark = SparkSession.builder.getOrCreate()
#   >>> profiler = DataProfiler(spark)
#   >>> df = spark.read.parquet("data/raw/synthetic_orders_with_pii.parquet")
#   >>> profile = profiler.profile_dataframe(df, "orders_raw")
#   >>> drift = profiler.detect_schema_drift(df, "config/schemas/orders.json")
#   >>> profiler.save_profile(profile, "docs/profiles/orders_raw.json")
# ============================================================================

from __future__ import annotations

import json
import re
import sys
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    NumericType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# PII regex patterns
# ---------------------------------------------------------------------------
_PII_PATTERNS: Dict[str, re.Pattern] = {
    "indian_phone": re.compile(
        r"(\+91[\-\s]?\d{5}\d{5})"
        r"|(\b0\d{2,4}[\-\s]?\d{6,8}\b)"
    ),
    "email": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),
    "aadhaar": re.compile(
        r"\b\d{4}\s\d{4}\s\d{4}\b"
    ),
    "pan_card": re.compile(
        r"\b[A-Z]{5}\d{4}[A-Z]\b"
    ),
    "credit_card": re.compile(
        r"\b(?:\d[ \-]*?){13,19}\b"
    ),
    "ipv4": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
}

# Spark SQL regex mirrors (used for DataFrame-level scans)
_PII_SPARK_PATTERNS: Dict[str, str] = {
    "indian_phone": r"(\+91[\- ]?\d{5}\d{5})|(0\d{2,4}[\- ]?\d{6,8})",
    "email": r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
    "aadhaar": r"\d{4} \d{4} \d{4}",
    "pan_card": r"[A-Z]{5}\d{4}[A-Z]",
}

# Numeric PySpark types (for branch logic)
_NUMERIC_TYPES: Tuple = (
    IntegerType,
    LongType,
    ShortType,
    FloatType,
    DoubleType,
)

# Percentile quantiles used for numeric profiling
_PERCENTILES: List[float] = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

# Maximum number of top-N values to return for categorical columns
_TOP_N: int = 20


# ============================================================================
# DataProfiler
# ============================================================================
class DataProfiler:
    """Profile PySpark DataFrames for quality, statistics, and PII exposure.

    The profiler produces a comprehensive dictionary describing every column in
    the DataFrame, including null counts, distinct value counts, data-type
    metadata, numeric descriptive statistics (with configurable percentiles),
    categorical frequency analysis, and potential PII detection via regex
    pattern matching.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        An active Spark session used for DataFrame operations.

    Attributes
    ----------
    spark : SparkSession
        The Spark session provided at construction time.

    Examples
    --------
    >>> spark = SparkSession.builder.getOrCreate()
    >>> profiler = DataProfiler(spark)
    >>> df = spark.read.parquet("data/raw/synthetic_orders_with_pii.parquet")
    >>> profile = profiler.profile_dataframe(df, "orders_raw")
    >>> print(profile["table_name"], profile["row_count"])
    orders_raw 50000
    """

    # ------------------------------------------------------------------ init
    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark

        # ---- Loguru setup -------------------------------------------------
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.remove()
        logger.add(
            sys.stderr,
            level="INFO",
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level:<8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
                " - <level>{message}</level>"
            ),
        )
        logger.add(
            str(log_dir / "data_profiling.log"),
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            level="DEBUG",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | "
                "{name}:{function}:{line} - {message}"
            ),
        )
        logger.info(
            "DataProfiler initialised  ·  Spark app = {}",
            spark.sparkContext.appName,
        )

    # ====================================================================
    # 1.  profile_dataframe
    # ====================================================================
    def profile_dataframe(
        self, df: DataFrame, table_name: str
    ) -> Dict[str, Any]:
        """Generate a comprehensive profile of *df*.

        The returned dictionary contains the following top-level keys:

        * ``table_name`` – echo of the *table_name* argument.
        * ``profile_timestamp`` – ISO-8601 UTC timestamp.
        * ``row_count`` – total number of rows.
        * ``column_count`` – total number of columns.
        * ``schema`` – list of ``{"name", "type", "nullable"}`` dicts.
        * ``columns`` – per-column profile (see below).
        * ``pii_summary`` – columns flagged as potential PII carriers.

        Per-column profile keys (inside ``columns[<col>]``):

        * ``data_type`` – Spark SQL type string.
        * ``nullable`` – whether the schema allows nulls.
        * ``null_count`` / ``null_percentage`` – absolute and relative nulls.
        * ``distinct_count`` / ``distinct_percentage`` – unique values.
        * ``is_numeric`` – True when the column is a numeric Spark type.
        * ``numeric_stats`` *(numeric only)* – ``min``, ``max``, ``mean``,
          ``stddev``, ``sum``, ``variance``, ``skewness``, ``kurtosis``, and
          ``percentiles`` (a dict keyed by quantile label).
        * ``is_categorical`` – True when the column is ``StringType`` with
          cardinality ≤ 100.
        * ``categorical_stats`` *(categorical only)* – ``cardinality``,
          ``top_values`` (list of ``{"value", "count", "percentage"}``),
          ``mode``, ``mode_frequency``.
        * ``min_length`` / ``max_length`` / ``avg_length`` *(string only)*.
        * ``pii_flags`` – dict mapping PII type → matched row count (> 0).

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The DataFrame to profile.
        table_name : str
            A human-readable label embedded in the profile output.

        Returns
        -------
        dict
            A JSON-serialisable profile dictionary.

        Raises
        ------
        RuntimeError
            If profiling fails (e.g. corrupt DataFrame).
        """
        logger.info(
            "Profiling table '{}' ({} columns) …", table_name, len(df.columns)
        )

        try:
            row_count: int = df.count()
            logger.debug("  row_count = {}", row_count)

            # ---- schema snapshot ------------------------------------------
            schema_snapshot: List[Dict[str, Any]] = [
                {
                    "name": field.name,
                    "type": str(field.dataType),
                    "nullable": field.nullable,
                }
                for field in df.schema.fields
            ]

            # ---- per-column profiling -------------------------------------
            columns_profile: Dict[str, Dict[str, Any]] = OrderedDict()

            for field in df.schema.fields:
                col_name: str = field.name
                logger.debug("  profiling column '{}'", col_name)

                col_profile: Dict[str, Any] = {
                    "data_type": str(field.dataType),
                    "nullable": field.nullable,
                }

                # ---- null analysis ----------------------------------------
                null_count = df.filter(F.col(col_name).isNull()).count()
                col_profile["null_count"] = null_count
                col_profile["null_percentage"] = round(
                    (null_count / row_count * 100) if row_count > 0 else 0.0, 4
                )

                # ---- distinct count ---------------------------------------
                distinct_count = df.select(col_name).distinct().count()
                col_profile["distinct_count"] = distinct_count
                col_profile["distinct_percentage"] = round(
                    (distinct_count / row_count * 100) if row_count > 0 else 0.0, 4
                )

                # ---- numeric branch ---------------------------------------
                is_numeric = isinstance(field.dataType, _NUMERIC_TYPES)
                col_profile["is_numeric"] = is_numeric

                if is_numeric:
                    col_profile["numeric_stats"] = self._compute_numeric_stats(
                        df, col_name, row_count
                    )

                # ---- categorical / string branch --------------------------
                is_string = isinstance(field.dataType, StringType)
                is_categorical = is_string and distinct_count <= 100
                col_profile["is_categorical"] = is_categorical

                if is_categorical:
                    col_profile["categorical_stats"] = (
                        self._compute_categorical_stats(
                            df, col_name, row_count
                        )
                    )

                # ---- string length stats ----------------------------------
                if is_string:
                    length_stats = self._compute_string_length_stats(
                        df, col_name
                    )
                    col_profile.update(length_stats)

                # ---- PII detection ----------------------------------------
                if is_string:
                    pii_flags = self._detect_pii_column(df, col_name)
                    col_profile["pii_flags"] = pii_flags
                else:
                    col_profile["pii_flags"] = {}

                columns_profile[col_name] = col_profile

            # ---- PII summary (columns with at least one flag) -------------
            pii_summary: Dict[str, List[str]] = {}
            for col_name, col_profile in columns_profile.items():
                flagged = [
                    pii_type
                    for pii_type, count in col_profile.get("pii_flags", {}).items()
                    if count > 0
                ]
                if flagged:
                    pii_summary[col_name] = flagged

            profile: Dict[str, Any] = {
                "table_name": table_name,
                "profile_timestamp": datetime.utcnow().isoformat(),
                "row_count": row_count,
                "column_count": len(df.columns),
                "schema": schema_snapshot,
                "columns": columns_profile,
                "pii_summary": pii_summary,
            }

            logger.success(
                "Profiling complete for '{}': {} rows, {} cols, {} PII columns",
                table_name,
                row_count,
                len(df.columns),
                len(pii_summary),
            )
            return profile

        except Exception as exc:
            logger.error(
                "Profiling failed for '{}': {}\n{}",
                table_name,
                exc,
                traceback.format_exc(),
            )
            raise RuntimeError(
                f"profile_dataframe failed for table '{table_name}'."
            ) from exc

    # ====================================================================
    # 2.  detect_schema_drift
    # ====================================================================
    def detect_schema_drift(
        self,
        current_df: DataFrame,
        reference_schema_path: str,
    ) -> Dict[str, Any]:
        """Compare the schema of *current_df* against a saved JSON reference.

        The reference file is expected to be a JSON object with a top-level
        ``"schema"`` key whose value is a list of
        ``{"name": str, "type": str, "nullable": bool}`` dicts – the same
        format produced by :meth:`profile_dataframe`.

        The drift report identifies:

        * **Added columns** – present in *current_df* but missing from the
          reference.
        * **Removed columns** – present in the reference but missing from
          *current_df*.
        * **Type changes** – columns whose data type differs.
        * **Nullable changes** – columns whose ``nullable`` flag differs.

        Parameters
        ----------
        current_df : pyspark.sql.DataFrame
            The DataFrame whose schema is compared.
        reference_schema_path : str
            Path to a JSON file containing the reference schema.  If the file
            does not exist, the current schema is saved as the new reference
            and an empty drift report is returned.

        Returns
        -------
        dict
            A drift report with keys:

            * ``"has_drift"`` – bool
            * ``"reference_path"`` – str
            * ``"drift_detected_at"`` – ISO-8601 timestamp
            * ``"added_columns"`` – list of column name strings
            * ``"removed_columns"`` – list of column name strings
            * ``"type_changes"`` – list of ``{"column", "reference_type",
              "current_type"}``
            * ``"nullable_changes"`` – list of ``{"column",
              "reference_nullable", "current_nullable"}``
            * ``"summary"`` – human-readable summary string

        Raises
        ------
        RuntimeError
            If the reference file exists but cannot be parsed.
        """
        logger.info(
            "Detecting schema drift against '{}'", reference_schema_path
        )

        ref_path = Path(reference_schema_path)

        # ---- current schema snapshot --------------------------------------
        current_schema: Dict[str, Dict[str, Any]] = {
            field.name: {
                "type": str(field.dataType),
                "nullable": field.nullable,
            }
            for field in current_df.schema.fields
        }

        # ---- load or bootstrap reference ----------------------------------
        if not ref_path.exists():
            logger.warning(
                "Reference schema '{}' not found – saving current schema as "
                "new reference.",
                reference_schema_path,
            )
            self._save_reference_schema(current_df, ref_path)
            return {
                "has_drift": False,
                "reference_path": str(ref_path),
                "drift_detected_at": datetime.utcnow().isoformat(),
                "added_columns": [],
                "removed_columns": [],
                "type_changes": [],
                "nullable_changes": [],
                "summary": (
                    "No reference existed. Current schema saved as new "
                    "reference – no drift to report."
                ),
            }

        try:
            with open(ref_path, "r", encoding="utf-8") as fh:
                ref_data = json.load(fh)

            ref_schema_list: List[Dict[str, Any]] = ref_data.get("schema", [])
            ref_schema: Dict[str, Dict[str, Any]] = {
                col["name"]: {
                    "type": col["type"],
                    "nullable": col.get("nullable", True),
                }
                for col in ref_schema_list
            }
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.error(
                "Failed to parse reference schema '{}': {}",
                reference_schema_path,
                exc,
            )
            raise RuntimeError(
                f"Cannot parse reference schema at '{reference_schema_path}'."
            ) from exc

        # ---- compare ------------------------------------------------------
        current_names = set(current_schema.keys())
        ref_names = set(ref_schema.keys())

        added_columns: List[str] = sorted(current_names - ref_names)
        removed_columns: List[str] = sorted(ref_names - current_names)

        type_changes: List[Dict[str, str]] = []
        nullable_changes: List[Dict[str, Any]] = []

        for col_name in sorted(current_names & ref_names):
            cur = current_schema[col_name]
            ref = ref_schema[col_name]

            if cur["type"] != ref["type"]:
                type_changes.append(
                    {
                        "column": col_name,
                        "reference_type": ref["type"],
                        "current_type": cur["type"],
                    }
                )
                logger.warning(
                    "  TYPE CHANGE  '{}': {} → {}",
                    col_name,
                    ref["type"],
                    cur["type"],
                )

            if cur["nullable"] != ref["nullable"]:
                nullable_changes.append(
                    {
                        "column": col_name,
                        "reference_nullable": ref["nullable"],
                        "current_nullable": cur["nullable"],
                    }
                )
                logger.warning(
                    "  NULLABLE CHANGE  '{}': {} → {}",
                    col_name,
                    ref["nullable"],
                    cur["nullable"],
                )

        has_drift = bool(
            added_columns or removed_columns or type_changes or nullable_changes
        )

        # ---- summary string -----------------------------------------------
        parts: List[str] = []
        if added_columns:
            parts.append(f"{len(added_columns)} added column(s)")
            for c in added_columns:
                logger.info("  ADDED  '{}'", c)
        if removed_columns:
            parts.append(f"{len(removed_columns)} removed column(s)")
            for c in removed_columns:
                logger.info("  REMOVED  '{}'", c)
        if type_changes:
            parts.append(f"{len(type_changes)} type change(s)")
        if nullable_changes:
            parts.append(f"{len(nullable_changes)} nullable change(s)")

        summary = "; ".join(parts) if parts else "No schema drift detected."

        if has_drift:
            logger.warning("Schema drift detected: {}", summary)
        else:
            logger.info("No schema drift detected.")

        return {
            "has_drift": has_drift,
            "reference_path": str(ref_path),
            "drift_detected_at": datetime.utcnow().isoformat(),
            "added_columns": added_columns,
            "removed_columns": removed_columns,
            "type_changes": type_changes,
            "nullable_changes": nullable_changes,
            "summary": summary,
        }

    # ====================================================================
    # 3.  save_profile
    # ====================================================================
    def save_profile(
        self,
        profile: Dict[str, Any],
        output_path: str,
    ) -> str:
        """Persist a profile dict as JSON and generate summary charts.

        The method writes two artefacts:

        1. A timestamped **JSON snapshot** at *output_path*.
        2. A set of **matplotlib / seaborn charts** in ``docs/profiles/``:
           * ``null_percentages.png`` – horizontal bar chart of null %.
           * ``distinct_counts.png`` – horizontal bar chart of distinct counts.
           * ``numeric_distributions.png`` – box-style summary per numeric col.
           * ``categorical_top_values.png`` – grouped bar chart of top values
             for every categorical column (max 5 columns shown).
           * ``pii_exposure.png`` – heatmap of PII type × column.

        Parameters
        ----------
        profile : dict
            The profile dictionary returned by :meth:`profile_dataframe`.
        output_path : str
            Destination path for the JSON file.  Parent directories are
            created automatically.

        Returns
        -------
        str
            The absolute path of the saved JSON file.

        Raises
        ------
        RuntimeError
            If saving or chart generation fails.
        """
        logger.info("Saving profile to '{}' …", output_path)

        try:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            # ---- write JSON -----------------------------------------------
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(profile, fh, indent=2, default=str)

            abs_path = str(out.resolve())
            logger.success("Profile JSON saved → {}", abs_path)

            # ---- generate charts ------------------------------------------
            charts_dir = Path("docs") / "profiles"
            charts_dir.mkdir(parents=True, exist_ok=True)

            table_name = profile.get("table_name", "unknown")

            self._chart_null_percentages(profile, charts_dir, table_name)
            self._chart_distinct_counts(profile, charts_dir, table_name)
            self._chart_numeric_distributions(profile, charts_dir, table_name)
            self._chart_categorical_top_values(profile, charts_dir, table_name)
            self._chart_pii_exposure(profile, charts_dir, table_name)

            logger.success(
                "Charts saved to {}/", charts_dir.resolve()
            )
            return abs_path

        except Exception as exc:
            logger.error(
                "save_profile failed: {}\n{}", exc, traceback.format_exc()
            )
            raise RuntimeError("save_profile failed.") from exc

    # ====================================================================
    # Internal helpers – statistics
    # ====================================================================
    def _compute_numeric_stats(
        self, df: DataFrame, col_name: str, row_count: int
    ) -> Dict[str, Any]:
        """Compute descriptive statistics for a numeric column.

        Parameters
        ----------
        df : DataFrame
            Source DataFrame.
        col_name : str
            Name of the numeric column.
        row_count : int
            Total row count (used for context, not computation).

        Returns
        -------
        dict
            Keys: ``min``, ``max``, ``mean``, ``stddev``, ``sum``,
            ``variance``, ``skewness``, ``kurtosis``, ``percentiles``.
        """
        stats_row = df.select(
            F.min(F.col(col_name)).alias("min_val"),
            F.max(F.col(col_name)).alias("max_val"),
            F.mean(F.col(col_name)).alias("mean_val"),
            F.stddev(F.col(col_name)).alias("stddev_val"),
            F.sum(F.col(col_name)).alias("sum_val"),
            F.variance(F.col(col_name)).alias("variance_val"),
            F.skewness(F.col(col_name)).alias("skewness_val"),
            F.kurtosis(F.col(col_name)).alias("kurtosis_val"),
        ).first()

        # Percentiles via approxQuantile
        try:
            percentile_values = df.stat.approxQuantile(
                col_name, _PERCENTILES, 0.01
            )
            percentiles = {
                f"p{int(q * 100):02d}": round(v, 4) if v is not None else None
                for q, v in zip(_PERCENTILES, percentile_values)
            }
        except Exception:
            percentiles = {}

        def _safe(val: Any) -> Any:
            """Convert NaN / None to None for JSON serialisation."""
            if val is None:
                return None
            try:
                import math
                if math.isnan(val) or math.isinf(val):
                    return None
            except (TypeError, ValueError):
                pass
            return round(float(val), 6)

        return {
            "min": _safe(stats_row["min_val"]),
            "max": _safe(stats_row["max_val"]),
            "mean": _safe(stats_row["mean_val"]),
            "stddev": _safe(stats_row["stddev_val"]),
            "sum": _safe(stats_row["sum_val"]),
            "variance": _safe(stats_row["variance_val"]),
            "skewness": _safe(stats_row["skewness_val"]),
            "kurtosis": _safe(stats_row["kurtosis_val"]),
            "percentiles": percentiles,
        }

    def _compute_categorical_stats(
        self, df: DataFrame, col_name: str, row_count: int
    ) -> Dict[str, Any]:
        """Compute frequency statistics for a categorical (string) column.

        Parameters
        ----------
        df : DataFrame
            Source DataFrame.
        col_name : str
            Name of the categorical column.
        row_count : int
            Total row count (used for percentage computation).

        Returns
        -------
        dict
            Keys: ``cardinality``, ``top_values``, ``mode``,
            ``mode_frequency``.
        """
        freq_df = (
            df.filter(F.col(col_name).isNotNull())
            .groupBy(col_name)
            .agg(F.count("*").alias("cnt"))
            .orderBy(F.desc("cnt"))
            .limit(_TOP_N)
        )
        freq_rows = freq_df.collect()

        top_values: List[Dict[str, Any]] = [
            {
                "value": row[col_name],
                "count": row["cnt"],
                "percentage": round(row["cnt"] / row_count * 100, 4)
                if row_count > 0
                else 0.0,
            }
            for row in freq_rows
        ]

        cardinality = df.select(col_name).distinct().count()
        mode = top_values[0]["value"] if top_values else None
        mode_frequency = top_values[0]["count"] if top_values else 0

        return {
            "cardinality": cardinality,
            "top_values": top_values,
            "mode": mode,
            "mode_frequency": mode_frequency,
        }

    def _compute_string_length_stats(
        self, df: DataFrame, col_name: str
    ) -> Dict[str, Any]:
        """Compute min / max / avg string length for a text column.

        Parameters
        ----------
        df : DataFrame
            Source DataFrame.
        col_name : str
            Name of the string column.

        Returns
        -------
        dict
            Keys: ``min_length``, ``max_length``, ``avg_length``.
        """
        length_row = (
            df.filter(F.col(col_name).isNotNull())
            .select(
                F.min(F.length(F.col(col_name))).alias("min_len"),
                F.max(F.length(F.col_name))).alias("max_len"),
                F.avg(F.length(F.col(col_name))).alias("avg_len"),
            )
            .first()
        )

        return {
            "min_length": int(length_row["min_len"])
            if length_row["min_len"] is not None
            else None,
            "max_length": int(length_row["max_len"])
            if length_row["max_len"] is not None
            else None,
            "avg_length": round(float(length_row["avg_len"]), 2)
            if length_row["avg_len"] is not None
            else None,
        }

    def _detect_pii_column(
        self, df: DataFrame, col_name: str
    ) -> Dict[str, int]:
        """Scan a string column for PII patterns using Spark SQL regex.

        Parameters
        ----------
        df : DataFrame
            Source DataFrame.
        col_name : str
            Name of the string column to scan.

        Returns
        -------
        dict
            Mapping of PII type name → count of matching rows.  Only types
            with count > 0 are included.
        """
        flags: Dict[str, int] = {}  

        for pii_type, pattern in _PII_SPARK_PATTERNS.items():
            try:
                match_count = (
                    df.filter(
                        F.col(col_name).isNotNull()
                        & F.col(col_name).rlike(pattern)
                    ).count()
                )
                if match_count > 0:
                    flags[pii_type] = match_count
                    logger.debug(
                        "    PII '{}' in '{}': {} rows",
                        pii_type,
                        col_name,
                        match_count,
                    )
            except Exception as exc:
                logger.warning(
                    "    PII scan for '{}' on '{}' failed: {}",
                    pii_type,
                    col_name,
                    exc,
                )

        return flags

    # ====================================================================
    # Internal helpers – schema reference
    # ====================================================================
    def _save_reference_schema(
        self, df: DataFrame, ref_path: Path
    ) -> None:
        """Save the current DataFrame schema as a JSON reference file.

        Parameters
        ----------
        df : DataFrame
            Source DataFrame.
        ref_path : pathlib.Path
            Destination path for the JSON reference.
        """
        ref_path.parent.mkdir(parents=True, exist_ok=True)

        schema_data = {
            "saved_at": datetime.utcnow().isoformat(),
            "schema": [
                {
                    "name": field.name,
                    "type": str(field.dataType),
                    "nullable": field.nullable,
                }
                for field in df.schema.fields
            ],
        }

        with open(ref_path, "w", encoding="utf-8") as fh:
            json.dump(schema_data, fh, indent=2)

        logger.info("Reference schema saved → {}", ref_path)

    # ====================================================================
    # Internal helpers – chart generation
    # ====================================================================
    def _chart_null_percentages(
        self,
        profile: Dict[str, Any],
        charts_dir: Path,
        table_name: str,
    ) -> None:
        """Horizontal bar chart of null percentages per column.

        Parameters
        ----------
        profile : dict
            Full profile dictionary.
        charts_dir : pathlib.Path
            Output directory for the chart image.
        table_name : str
            Used in the chart title.
        """
        columns = profile.get("columns", {})
        names = list(columns.keys())
        nulls = [columns[n].get("null_percentage", 0) for n in names]

        if not names:
            return

        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4)))
        colors = ["#e74c3c" if v > 5 else "#2ecc71" for v in nulls]
        ax.barh(names, nulls, color=colors)
        ax.set_xlabel("Null %")
        ax.set_title(f"Null Percentages – {table_name}")
        ax.invert_yaxis()

        for i, v in enumerate(nulls):
            ax.text(v + 0.3, i, f"{v:.2f}%, va='center', fontsize=8")

        plt.tight_layout()
        path = charts_dir / f"{table_name}_null_percentages.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        logger.debug("  chart → {}", path)

    def _chart_distinct_counts(
        self,
        profile: Dict[str, Any],
        charts_dir: Path,
        table_name: str,
    ) -> None:
        """Horizontal bar chart of distinct value counts per column.

        Parameters
        ----------
        profile : dict
            Full profile dictionary.
        charts_dir : pathlib.Path
            Output directory for the chart image.
        table_name : str
            Used in the chart title.
        """
        columns = profile.get("columns", {})
        names = list(columns.keys())
        distincts = [columns[n].get("distinct_count", 0) for n in names]

        if not names:
            return

        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4)))
        ax.barh(names, distincts, color="#3498db")
        ax.set_xlabel("Distinct Count")
        ax.set_title(f"Distinct Value Counts – {table_name}")
        ax.invert_yaxis()

        for i, v in enumerate(distincts):
            ax.text(v + 0.5, i, f"{v:,}", va="center", fontsize=8)

        plt.tight_layout()
        path = charts_dir / f"{table_name}_distinct_counts.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        logger.debug("  chart → {}", path)

    def _chart_numeric_distributions(
        self,
        profile: Dict[str, Any],
        charts_dir: Path,
        table_name: str,
    ) -> None:
        """Box-style summary chart for numeric columns.

        For each numeric column the chart shows min, Q1, median, Q3, max
        as a horizontal span with markers.

        Parameters
        ----------
        profile : dict
            Full profile dictionary.
        charts_dir : pathlib.Path
            Output directory for the chart image.
        table_name : str
            Used in the chart title.
        """
        columns = profile.get("columns", {})
        numeric_cols: List[str] = [
            n for n, c in columns.items() if c.get("is_numeric", False)
        ]

        if not numeric_cols:
            logger.debug("  no numeric columns – skipping distribution chart")
            return

        fig, ax = plt.subplots(
            figsize=(12, max(3, len(numeric_cols) * 0.8))
        )

        y_positions: List[int] = []
        y_labels: List[str] = []

        for i, col_name in enumerate(numeric_cols):
            stats = columns[col_name].get("numeric_stats", {})
            percentiles = stats.get("percentiles", {})

            min_val = stats.get("min")
            max_val = stats.get("max")
            mean_val = stats.get("mean")
            q1 = percentiles.get("p25")
            median = percentiles.get("p50")
            q3 = percentiles.get("p75")

            vals = [v for v in [min_val, q1, median, q3, max_val] if v is not None]
            if not vals:
                continue

            y_pos = i
            y_positions.append(y_pos)
            y_labels.append(col_name)

            # whisker line (min to max)
            if min_val is not None and max_val is not None:
                ax.plot(
                    [min_val, max_val], [y_pos, y_pos],
                    color="#95a5a6", linewidth=1.5, zorder=1,
                )

            # IQR box
            if q1 is not None and q3 is not None:
                ax.barh(
                    y_pos, q3 - q1, left=q1, height=0.4,
                    color="#3498db", alpha=0.6, zorder=2,
                )

            # median marker
            if median is not None:
                ax.plot(
                    median, y_pos, "D", color="#e74c3c",
                    markersize=6, zorder=3,
                )

            # mean marker
            if mean_val is not None:
                ax.plot(
                    mean_val, y_pos, "^", color="#2ecc71",
                    markersize=6, zorder=3,
                )

        if not y_positions:
            plt.close(fig)
            return

        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Value")
        ax.set_title(
            f"Numeric Distributions – {table_name}  "
            "(◆ median, ▲ mean)"
        )
        ax.invert_yaxis()
        plt.tight_layout()

        path = charts_dir / f"{table_name}_numeric_distributions.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        logger.debug("  chart → {}", path)

    def _chart_categorical_top_values(
        self,
        profile: Dict[str, Any],
        charts_dir: Path,
        table_name: str,
    ) -> None:
        """Grouped bar chart showing top values for categorical columns.

        At most 5 categorical columns are shown (sorted by cardinality
        ascending so that columns with few distinct values appear first).

        Parameters
        ----------
        profile : dict
            Full profile dictionary.
        charts_dir : pathlib.Path
            Output directory for the chart image.
        table_name : str
            Used in the chart title.
        """
        columns = profile.get("columns", {})
        cat_cols: List[Tuple[str, int]] = [
            (n, c.get("categorical_stats", {}).get("cardinality", 0))
            for n, c in columns.items()
            if c.get("is_categorical", False)
        ]
        cat_cols.sort(key=lambda x: x[1])

        if not cat_cols:
            logger.debug("  no categorical columns – skipping top-values chart")
            return

        # Limit to 5 columns for readability
        selected = [name for name, _ in cat_cols[:5]]

        n_cols = len(selected)
        fig, axes = plt.subplots(
            n_cols, 1,
            figsize=(10, max(4, n_cols * 3)),
            squeeze=False,
        )

        palette = sns.color_palette("viridis", _TOP_N)

        for idx, col_name in enumerate(selected):
            ax = axes[idx, 0]
            stats = columns[col_name].get("categorical_stats", {})
            top = stats.get("top_values", [])[:10]  # show top 10 per chart

            if not top:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center")
                ax.set_title(col_name)
                continue

            labels = [str(t["value"]) for t in top]
            counts = [t["count"] for t in top]

            ax.barh(labels, counts, color=palette[:len(labels)])
            ax.set_xlabel("Count")
            ax.set_title(f"{col_name}  (cardinality={stats.get('cardinality', '?')})")
            ax.invert_yaxis()

        plt.suptitle(f"Categorical Top Values – {table_name}", fontsize=14)
        plt.tight_layout()

        path = charts_dir / f"{table_name}_categorical_top_values.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        logger.debug("  chart → {}", path)

    def _chart_pii_exposure(
        self,
        profile: Dict[str, Any],
        charts_dir: Path,
        table_name: str,
    ) -> None:
        """Heatmap showing PII type exposure across columns.

        Rows = columns with at least one PII flag, Columns = PII types.
        Cell values = row counts matching each PII pattern.

        Parameters
        ----------
        profile : dict
            Full profile dictionary.
        charts_dir : pathlib.Path
            Output directory for the chart image.
        table_name : str
            Used in the chart title.
        """
        pii_summary = profile.get("pii_summary", {})
        if not pii_summary:
            logger.debug("  no PII flags – skipping PII heatmap")
            return

        all_pii_types: List[str] = sorted(
            {ptype for ptypes in pii_summary.values() for ptype in ptypes}
        )
        col_names = sorted(pii_summary.keys())

        columns_data = profile.get("columns", {})
        matrix: List[List[int]] = []

        for col_name in col_names:
            row: List[int] = []
            flags = columns_data.get(col_name, {}).get("pii_flags", {})
            for pii_type in all_pii_types:
                row.append(flags.get(pii_type, 0))
            matrix.append(row)

        fig, ax = plt.subplots(
            figsize=(max(6, len(all_pii_types) * 1.8), max(3, len(col_names) * 0.6))
        )

        sns.heatmap(
            matrix,
            annot=True,
            fmt=",d",
            cmap="YlOrRd",
            xticklabels=all_pii_types,
            yticklabels=col_names,
            linewidths=0.5,
            ax=ax,
        )

        ax.set_title(f"PII Exposure Heatmap – {table_name}")
        ax.set_xlabel("PII Type")
        ax.set_ylabel("Column")
        plt.tight_layout()

        path = charts_dir / f"{table_name}_pii_exposure.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        logger.debug("  chart → {}", path)


# ============================================================================
# CLI entry-point / smoke test
# ============================================================================
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("DataProfiler – smoke test")
    logger.info("=" * 70)

    try:
        spark = (
            SparkSession.builder.appName("DataProfiler-SmokeTest")
            .config(
                "spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension",
            )
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .config(
                "spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0"
            )
            .getOrCreate()
        )

        profiler = DataProfiler(spark)

        raw_path = "data/raw/synthetic_orders_with_pii.parquet"
        if Path(raw_path).exists():
            df = spark.read.parquet(raw_path)
            logger.info("Loaded {} records from '{}'.", df.count(), raw_path)

            # 1. Profile
            profile = profiler.profile_dataframe(df, "orders_raw")
            logger.info(
                "Profile keys: {}", list(profile.keys())
            )

            # 2. Schema drift
            drift = profiler.detect_schema_drift(
                df, "config/schemas/orders_reference.json"
            )
            logger.info("Drift detected: {}", drift["has_drift"])

            # 3. Save profile + charts
            saved_path = profiler.save_profile(
                profile, "docs/profiles/orders_raw.json"
            )
            logger.success("Saved profile → {}", saved_path)
        else:
            logger.warning(
                "Sample data not found at '{}'. "
                "Run src/ingestion/data_collector.py first.",
                raw_path,
            )

        spark.stop()
        logger.success("Smoke test completed.")

    except Exception as exc:
        logger.exception("Smoke test failed: {}", exc)
        sys.exit(1)