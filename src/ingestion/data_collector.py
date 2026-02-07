# ============================================================================
# Adaptive Data Governance Framework
# src/ingestion/data_collector.py
# ============================================================================
# DataCollector – generates synthetic e-commerce order data laced with
# realistic PII patterns (Indian locale) and optionally injects common
# data-quality issues so the downstream governance pipeline has something
# meaningful to detect and remediate.
# ============================================================================

from __future__ import annotations

import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRODUCT_CATEGORIES: List[str] = [
    "Electronics",
    "Fashion",
    "Home",
    "Books",
    "Beauty",
]

PAYMENT_METHODS: List[str] = [
    "UPI",
    "Card",
    "COD",
    "Wallet",
    "NetBanking",
]

INDIAN_CITIES: List[str] = [
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Hyderabad",
    "Chennai",
    "Kolkata",
    "Pune",
    "Ahmedabad",
    "Jaipur",
    "Lucknow",
    "Chandigarh",
    "Bhopal",
    "Patna",
    "Indore",
    "Nagpur",
    "Coimbatore",
    "Kochi",
    "Visakhapatnam",
    "Thiruvananthapuram",
    "Guwahati",
]

# Percentage thresholds for PII injection into delivery_instructions
_PII_INJECTION_RATE: float = 0.60  # 60 % of rows contain hidden PII


# ============================================================================
# DataCollector
# ============================================================================
class DataCollector:
    """Generate and persist synthetic e-commerce order data with embedded PII.

    Parameters
    ----------
    raw_data_path : str
        Directory where raw Parquet files are written.  Created automatically
        if it does not already exist.
    """

    # ------------------------------------------------------------------ init
    def __init__(self, raw_data_path: str = "./data/raw") -> None:
        self.raw_data_path = Path(raw_data_path)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        # Loguru – rotating file sink
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.remove()  # remove default stderr handler
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>",
        )
        logger.add(
            str(log_dir / "data_collection.log"),
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | "
            "{name}:{function}:{line} - {message}",
        )

        self.fake = Faker("en_IN")
        Faker.seed(42)
        random.seed(42)
        np.random.seed(42)

        logger.info(
            "DataCollector initialised  ·  raw_data_path={}", self.raw_data_path
        )

    # ------------------------------------------------- internal PII helpers
    def _random_aadhaar(self) -> str:
        """Return a fake 12-digit Aadhaar-style number."""
        return " ".join(
            [
                str(random.randint(1000, 9999)),
                str(random.randint(1000, 9999)),
                str(random.randint(1000, 9999)),
            ]
        )

    def _random_indian_phone(self) -> str:
        """Return a fake Indian mobile number."""
        return f"+91-{random.randint(70000, 99999)}{random.randint(10000, 99999)}"

    def _inject_pii_into_text(self, base_text: str) -> str:
        """Randomly embed one or more PII fragments into *base_text*.

        The injected PII is chosen uniformly from: phone number, full name,
        Aadhaar number, or e-mail address.
        """
        pii_options = [
            f"Call me at {self._random_indian_phone()}",
            f"Contact {self.fake.name()} for delivery",
            f"Aadhaar: {self._random_aadhaar()}",
            f"Email me at {self.fake.email()}",
            f"My number is {self._random_indian_phone()}, call before delivery",
            f"Send OTP to {self.fake.email()} or {self._random_indian_phone()}",
        ]
        pii_snippet = random.choice(pii_options)
        # Insert the PII snippet at a natural-looking position
        return f"{base_text}. {pii_snippet}"

    # ----------------------------------------- generate_synthetic_pii_data
    def generate_synthetic_pii_data(
        self, n_records: int = 50_000
    ) -> pd.DataFrame:
        """Create *n_records* synthetic e-commerce orders with embedded PII.

        Approximately 60 % of ``delivery_instructions`` rows will contain
        hidden PII (phone numbers, names, Aadhaar numbers, or e-mail
        addresses).  The remaining 40 % are clean.

        Parameters
        ----------
        n_records : int
            Number of order rows to generate.

        Returns
        -------
        pd.DataFrame
            The generated DataFrame (also persisted as Parquet).
        """
        logger.info("Generating {} synthetic e-commerce orders …", n_records)

        now = datetime.now()
        one_year_ago = now - timedelta(days=365)

        records: List[dict] = []
        for i in range(n_records):
            # -- base delivery instruction (clean) -------------------------
            base_instruction = self.fake.sentence(nb_words=8)

            # -- conditionally inject PII ----------------------------------
            if random.random() < _PII_INJECTION_RATE:
                delivery_instructions = self._inject_pii_into_text(
                    base_instruction
                )
            else:
                delivery_instructions = base_instruction

            record = {
                "order_id": str(uuid.uuid4()),
                "customer_id": str(uuid.uuid4()),
                "product_category": random.choice(PRODUCT_CATEGORIES),
                "order_value": round(random.uniform(500.0, 50_000.0), 2),
                "delivery_instructions": delivery_instructions,
                "customer_review": self.fake.text(max_nb_chars=200),
                "order_timestamp": self.fake.date_time_between(
                    start_date=one_year_ago, end_date=now
                ),
                "delivery_city": random.choice(INDIAN_CITIES),
                "delivery_pincode": str(random.randint(100_000, 999_999)),
                "payment_method": random.choice(PAYMENT_METHODS),
            }
            records.append(record)

            # progress logging every 10 000 rows
            if (i + 1) % 10_000 == 0:
                logger.debug("  … generated {}/{} records", i + 1, n_records)

        df = pd.DataFrame(records)

        # -- persist --------------------------------------------------------
        output_path = self.raw_data_path / "synthetic_orders_with_pii.parquet"
        df.to_parquet(output_path, index=False, engine="pyarrow")
        logger.success(
            "Saved {} records → {}", len(df), output_path
        )

        # -- quick stats ----------------------------------------------------
        pii_count = int(
            df["delivery_instructions"].str.contains(
                r"(\+91|Aadhaar|@|Call me|Contact)", regex=True
            ).sum()
        )
        logger.info(
            "PII prevalence: {}/{} rows ({:.1f} %)",
            pii_count,
            len(df),
            pii_count / len(df) * 100,
        )

        return df

    # ----------------------------------------- inject_data_quality_issues
    def inject_data_quality_issues(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Inject realistic data-quality issues into an existing DataFrame.

        Issue breakdown (applied to a *copy* of the input):
        * **Null values** – ~5 % of rows get NaN in random columns.
        * **Negative prices** – ~2 % of ``order_value`` entries are negated.
        * **Duplicate order_ids** – ~3 % of rows are duplicated.
        * **Future dates** – ~1 % of ``order_timestamp`` entries are set to a
          random date in the next year.

        Parameters
        ----------
        df : pd.DataFrame
            The clean (or PII-injected) DataFrame to degrade.

        Returns
        -------
        pd.DataFrame
            A new DataFrame containing all original + degraded rows.
        """
        logger.info(
            "Injecting data-quality issues into {} rows …", len(df)
        )
        dirty = df.copy()
        n = len(dirty)

        # 1. Null values – 5 %
        null_mask = np.random.random(n) < 0.05
        null_cols = [
            "customer_id",
            "delivery_city",
            "delivery_pincode",
            "payment_method",
        ]
        null_indices = dirty.index[null_mask]
        for idx in null_indices:
            col = random.choice(null_cols)
            dirty.at[idx, col] = None
        logger.debug("  → injected NULLs in {} rows", int(null_mask.sum()))

        # 2. Negative prices – 2 %
        neg_mask = np.random.random(n) < 0.02
        dirty.loc[neg_mask, "order_value"] = (
            dirty.loc[neg_mask, "order_value"] * -1
        )
        logger.debug(
            "  → negated order_value in {} rows", int(neg_mask.sum())
        )

        # 3. Duplicate order_ids – 3 %
        dup_mask = np.random.random(n) < 0.03
        dup_rows = dirty.loc[dup_mask].copy()
        dirty = pd.concat([dirty, dup_rows], ignore_index=True)
        logger.debug(
            "  → duplicated {} rows (total now {})", len(dup_rows), len(dirty)
        )

        # 4. Future dates – 1 %
        future_mask = np.random.random(len(dirty)) < 0.01
        future_dates = [
            datetime.now() + timedelta(days=random.randint(1, 365))
            for _ in range(int(future_mask.sum()))
        ]
        dirty.loc[future_mask, "order_timestamp"] = future_dates
        logger.debug(
            "  → set future timestamps in {} rows", int(future_mask.sum()))

        # -- persist --------------------------------------------------------
        output_path = (
            self.raw_data_path / "synthetic_orders_with_issues.parquet"
        )
        dirty.to_parquet(output_path, index=False, engine="pyarrow")
        logger.success(
            "Saved {} records (with issues) → {}", len(dirty), output_path
        )

        return dirty

    # ----------------------------------------- fetch_olist_instructions
    def fetch_olist_instructions(self) -> None:
        """Log instructions for downloading the Olist e-commerce dataset.

        The Olist Brazilian E-Commerce dataset is publicly available on
        Kaggle.  This method prints the ``kaggle`` CLI command required to
        download it into the project's ``data/raw`` directory.
        """
        logger.info("=" * 70)
        logger.info("OLIST DATASET DOWNLOAD INSTRUCTIONS")
        logger.info("=" * 70)
        logger.info(
            "The Olist Brazilian E-Commerce Public Dataset is available on "
            "Kaggle at:"
        )
        logger.info(
            "  https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
        )
        logger.info("")
        logger.info("To download via the Kaggle CLI, run:")
        logger.info(
            "  kaggle datasets download -d olistbr/brazilian-ecommerce "
            f"-p {self.raw_data_path} --unzip"
        )
        logger.info("")
        logger.info("Prerequisites:")
        logger.info("  1. pip install kaggle")
        logger.info(
            "  2. Place your kaggle.json API token in ~/.kaggle/kaggle.json"
        )
        logger.info(
            "  3. chmod 600 ~/.kaggle/kaggle.json"
        )
        logger.info("=" * 70)


# ============================================================================
# CLI entry-point
# ============================================================================
if __name__ == "__main__":
    logger.info("Starting synthetic data generation pipeline …")

    try:
        collector = DataCollector()

        # Step 1 – generate clean data with PII
        clean_df = collector.generate_synthetic_pii_data(n_records=50_000)

        # Step 2 – inject data-quality issues
        dirty_df = collector.inject_data_quality_issues(clean_df)

        # Step 3 – print Olist download instructions
        collector.fetch_olist_instructions()

        logger.success("Data generation pipeline completed successfully.")
        logger.info(
            "Clean dataset : {} rows", len(clean_df)
        )
        logger.info(
            "Dirty dataset : {} rows", len(dirty_df)
        )

    except Exception as exc:
        logger.exception("Data generation pipeline failed: {}", exc)
        sys.exit(1)