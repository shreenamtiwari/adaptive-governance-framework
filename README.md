# Adaptive Data Governance Framework

> **Dissertation Project** — An AI-driven, self-improving data governance framework for Indian e-commerce platforms, built with PySpark, Delta Lake, and Apache Airflow.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [AI / ML Models](#ai--ml-models)
- [Pipeline Flow](#pipeline-flow)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Access Points](#access-points)
- [Pipeline Output](#pipeline-output)
- [Configuration](#configuration)
- [Key Modules](#key-modules)

---

## Overview

This framework implements a production-grade **Adaptive Data Governance** system for e-commerce data platforms. Unlike traditional rule-based governance, every threshold, weight, and detection boundary **learns and adapts** from historical pipeline runs.

### What Makes It Adaptive

| Capability | Traditional | This Framework |
|---|---|---|
| DQ Pass/Fail Threshold | Hard-coded (e.g. 85%) | Learned from rolling baseline (μ − kσ) |
| DQ Dimension Weights | Equal (20% each) | Inverse-mean + linear regression from history |
| Anomaly Detection | Single method | 3 methods: Z-score, IQR, Isolation Forest |
| PII Detection | Regex only | Regex + DistilBERT NER (dslim/bert-base-NER) |
| PII Confidence | Static threshold | F1-optimal per-entity-type tuning from feedback |
| PII Monitoring | None | Drift detection (FN rate baseline vs recent) |
| Quality Trends | None | Early warning system (3-run decline detection) |

### Key Technologies

| Component | Technology |
|---|---|
| Processing Engine | Apache PySpark 3.5.0 |
| Storage Layer | Delta Lake 3.0.0 (ACID, time travel) |
| Orchestration | Apache Airflow 2.8.0 |
| ML / AI | scikit-learn 1.3.2, Hugging Face Transformers 4.36.2, PyTorch 2.1.2 |
| NER Model | dslim/bert-base-NER (DistilBERT fine-tuned for NER) |
| Containerisation | Docker Compose (6 services) |
| Language | Python 3.10 |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Apache Airflow DAG (11 Tasks)                        │
│                                                                         │
│  start → generate_data → ingest_bronze ─┬─ streaming_ingestion ──┐     │
│                                          │                         │     │
│                                          └─ bronze_to_silver       │     │
│                                                │                   │     │
│                                          quality_gate (AI Engine)  │     │
│                                                │                   │     │
│                                          silver_to_gold            │     │
│                                                │                   │     │
│                                          pii_scan_summary ─────────┤     │
│                                                │                   │     │
│                                          log_completion ◄──────────┘     │
│                                                │                         │
│                                               end                        │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                     Medallion Architecture (Delta Lake)                  │
│                                                                         │
│  Raw/Streaming  ──►  Bronze (Delta)  ──►  Silver (Delta)  ──►  Gold     │
│  - 500K orders       - Schema drift       - PII masking        - CLV   │
│  - 103K customers      detection          - Quarantine         - RFM   │
│  - 10K products      - Append-only        - DQ validation      - Churn │
│  - 200K reviews      - Metadata           - Deduplication      - Rev   │
│  - 20K clickstream                                             - Golden│
│                                                                  Rec.  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐          │
│  │  Quarantine  │    │ Data Contract│    │ Streaming Bronze │          │
│  │  (Failed DQ) │    │ (YAML SLAs)  │    │ (Clickstream)    │          │
│  └──────────────┘    └──────────────┘    └──────────────────┘          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                     Adaptive AI Engine (12 Models)                       │
│                                                                         │
│  ┌───────────────────┐  ┌───────────────────┐  ┌────────────────────┐  │
│  │ Anomaly Detection │  │ Adaptive Scoring  │  │ PII Intelligence   │  │
│  │ - Z-Score         │  │ - Rolling thresh. │  │ - Regex (8 types)  │  │
│  │ - IQR Fences      │  │ - Weight learning │  │ - NER (DistilBERT) │  │
│  │ - Isolation Forest│  │ - Regression wts  │  │ - F1 tuner         │  │
│  │ - Batch anomaly   │  │ - Early warning   │  │ - Drift detection  │  │
│  └───────────────────┘  └───────────────────┘  └────────────────────┘  │
│                                                                         │
│  ┌───────────────────┐  ┌───────────────────┐  ┌────────────────────┐  │
│  │ Identity Resoln.  │  │ Data Contracts    │  │ Governance Reports │  │
│  │ - Jaro-Winkler    │  │ - YAML schemas    │  │ - JSON timestamped │  │
│  │ - Golden records  │  │ - SLA enforcement │  │ - Full audit trail │  │
│  └───────────────────┘  └───────────────────┘  └────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## AI / ML Models

Every model listed below is **actually invoked at runtime** during the pipeline execution:

### 1. Z-Score Anomaly Detection
- **File**: `src/quality/anomaly_detector.py` → `zscore_detect()`
- **Method**: Computes mean and stddev per numeric column; flags rows where |value − μ| > z_threshold × σ
- **Invoked by**: `AdaptiveGovernanceEngine.evaluate()` → Step 2a

### 2. IQR Fence Anomaly Detection
- **File**: `src/quality/anomaly_detector.py` → `iqr_detect()`
- **Method**: Computes Q1, Q3 via `approxQuantile`; flags rows outside [Q1 − 1.5×IQR, Q3 + 1.5×IQR]
- **Invoked by**: `AdaptiveGovernanceEngine.evaluate()` → Step 2b

### 3. Isolation Forest (sklearn)
- **File**: `src/quality/anomaly_detector.py` → `isolation_forest_detect()`
- **Method**: Trains `sklearn.ensemble.IsolationForest` on a 10% sample, scores all rows
- **Invoked by**: `AdaptiveGovernanceEngine.evaluate()` → Step 2c

### 4. Adaptive DQ Threshold
- **File**: `src/quality/adaptive_scorer.py` → `compute_adaptive_threshold()`
- **Method**: Rolling mean − k×std over last N runs; clamped to [70%, 99%]
- **Invoked by**: `AdaptiveGovernanceEngine.evaluate()` → Step 3

### 5. Dimension Weight Learning (Inverse-Mean)
- **File**: `src/quality/adaptive_scorer.py` → `learn_dimension_weights()`
- **Method**: Lower-scoring dimensions get higher weights (inverse-mean normalised)
- **Invoked by**: `AdaptiveGovernanceEngine.evaluate()` → Step 1

### 6. Dimension Weight Learning (Linear Regression)
- **File**: `src/quality/adaptive_scorer.py` → `learn_weights_regression()`
- **Method**: Fits sklearn LinearRegression to predict overall score from dimension scores
- **Invoked by**: `AdaptiveGovernanceEngine.evaluate()` → Step 9

### 7. Early Warning System
- **File**: `src/quality/adaptive_scorer.py` → `check_early_warning()`
- **Method**: Detects 3-run consecutive decline, below-mean scores, accelerating degradation
- **Invoked by**: `AdaptiveGovernanceEngine.evaluate()` → Step 4

### 8. Batch Anomaly Detection
- **File**: `src/quality/anomaly_detector.py` → `detect_batch_anomaly()`
- **Method**: Cross-run Z-score comparison using persisted DQ history
- **Invoked by**: `AdaptiveGovernanceEngine.evaluate()` → Step 5

### 9. PII Detection — Regex (8 Patterns)
- **File**: `src/pii_detection/pii_detector.py` → `detect_pii()`
- **Patterns**: EMAIL, PHONE_NUMBER, AADHAAR, PAN, CREDIT_CARD, IPV4, ADDRESS, IFSC
- **Invoked by**: `_pii_scan_summary()` task + `BronzeToSilverTransformer`

### 10. PII Detection — NER (DistilBERT)
- **File**: `src/pii_detection/pii_detector.py` → NER pipeline
- **Model**: `dslim/bert-base-NER` (Hugging Face) with chunked processing
- **Invoked by**: `_pii_scan_summary()` task with `use_ner_model=True`

### 11. PII Confidence Tuner
- **File**: `src/pii_detection/adaptive_pii_tuner.py` → `tune_thresholds()`
- **Method**: Grid search over 50 candidate thresholds to maximise F1 per entity type
- **Invoked by**: `_pii_scan_summary()` task + `AdaptiveGovernanceEngine.evaluate()` → Step 8

### 12. PII Drift Detection
- **File**: `src/pii_detection/adaptive_pii_tuner.py` → `detect_pii_drift()`
- **Method**: Compares false-negative rates between baseline (75%) and recent (25%) feedback
- **Invoked by**: `_pii_scan_summary()` task + `AdaptiveGovernanceEngine.evaluate()` → Step 7

---

## Pipeline Flow

The `medallion_pipeline_dag` runs **11 tasks** in the following order:

| # | Task | What It Does |
|---|---|---|
| 1 | `start` | Pipeline entry point |
| 2 | `generate_synthetic_data` | Generates 500K orders, 103K customers, 10K products, 200K reviews, 500K order items with real-world scenarios (fraud, festival spikes, PII leakage, duplicates) |
| 3 | `ingest_to_bronze` | Reads raw Parquet → writes to Bronze Delta Lake (5 tables) |
| 4 | `streaming_ingestion` | Produces 10 micro-batches × 2,000 clickstream events with 5% PII injection; consumes via Structured Streaming |
| 5 | `bronze_to_silver` | PII masking (hash/redact), quarantine invalid records, add Silver metadata |
| 6 | `data_quality_check` | **Adaptive AI Engine** — runs all 12 AI models: 3 anomaly detectors, adaptive threshold, weight learning, early warning, batch anomaly, PII drift, PII tuning |
| 7 | `silver_to_gold` | Revenue aggregates, RFM segmentation, CLV scoring, churn features, Identity Resolution (Jaro-Winkler dedup → golden records) |
| 8 | `pii_scan_summary` | Scans Silver with Regex + NER, records PII feedback, auto-tunes thresholds, checks drift |
| 9 | `log_completion` | Prints full pipeline summary with row counts, all governance metrics, and AI model execution log |
| 10 | `end` | Pipeline exit |

---

## Project Structure

```
adaptive-governance-framework/
├── README.md                              # This file
├── docker-compose.yml                     # 6 services: Spark, Airflow, Jupyter, PostgreSQL
├── Dockerfile.jupyter                     # JupyterLab with PySpark
├── requirements.txt                       # Host Python dependencies
├── requirements.airflow.txt               # Airflow container dependencies
├── requirements.jupyter.txt               # Jupyter container dependencies
│
├── airflow/dags/
│   └── medallion_pipeline_dag.py          # Main 11-task DAG
│
├── config/
│   ├── config.yaml                        # Central configuration
│   └── data_contracts/                    # YAML data contract definitions
│       └── ecommerce_orders_v2.0.0.yaml
│
├── scripts/
│   └── deploy.sh                          # One-command Docker deployment
│
├── src/
│   ├── governance/
│   │   ├── adaptive_governance_engine.py  # Central AI orchestrator (12 models)
│   │   ├── identity_resolution.py         # Jaro-Winkler fuzzy matching + golden records
│   │   └── data_contracts.py              # YAML data contracts + SLA enforcement
│   │
│   ├── quality/
│   │   ├── anomaly_detector.py            # Z-score, IQR, Isolation Forest
│   │   ├── adaptive_scorer.py             # Self-tuning thresholds + weight learning
│   │   └── quality_metrics.py             # 5 DQ dimensions
│   │
│   ├── pii_detection/
│   │   ├── pii_detector.py                # Regex + NER PII detection
│   │   ├── pii_masker.py                  # Hash / redact / tokenize masking
│   │   └── adaptive_pii_tuner.py          # F1-optimal threshold tuning + drift
│   │
│   ├── transformation/
│   │   ├── bronze_to_silver.py            # PII masking + quarantine + metadata
│   │   └── silver_to_gold.py              # CLV, RFM, churn, revenue aggregations
│   │
│   ├── ingestion/
│   │   ├── data_loader.py                 # Raw → Bronze with schema drift detection
│   │   ├── data_generator.py              # Large-scale synthetic Indian e-commerce data
│   │   └── streaming_simulator.py         # Micro-batch producer + Structured Streaming
│   │
│   └── utils/
│       ├── spark_utils.py                 # SparkSession builder with Delta Lake
│       └── schemas.py                     # Shared PySpark schemas
│
├── data/                                  # Generated at runtime (gitignored)
│   ├── raw/                               # Source Parquet files
│   ├── bronze/                            # Bronze Delta tables
│   ├── silver/                            # Silver Delta tables
│   ├── gold/                              # Gold Delta tables
│   ├── quarantine/                        # Failed DQ records
│   └── streaming/                         # Streaming landing zone
│
├── docs/                                  # Documentation
│   ├── deployment_guide.md                # Step-by-step deployment
│   ├── architecture.md                    # Architecture decisions
│   └── dpdp_compliance.md                 # DPDP Act 2023 compliance
│
├── tests/                                 # Unit + integration tests
├── notebooks/                             # Jupyter exploration
└── models/                                # Trained ML models
```

---

## Prerequisites

| Requirement | Minimum |
|---|---|
| macOS / Linux | macOS 12+ or Ubuntu 20.04+ |
| RAM | 16 GB (Docker needs 12 GB allocated) |
| Disk | 50 GB free |
| Docker Desktop | 4.25+ with Compose V2 |
| Docker Memory | 12 GB minimum (Settings → Resources) |

> **Note**: Python, Java, Spark are all containerised — no local installation required.

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/KartikayRaniwala/adaptive-governance-framework.git
cd adaptive-governance-framework
```

### 2. Configure Docker Resources

Open **Docker Desktop → Settings → Resources**:
- CPUs: 6+
- Memory: **12 GB minimum** (16 GB recommended)
- Swap: 4 GB

### 3. Deploy (One Command)

```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

This will:
1. Build all Docker images (Spark, Airflow, JupyterLab)
2. Start 6 services (PostgreSQL, Spark Master/Worker, Airflow, JupyterLab)
3. Wait for health checks
4. Clear stale data and caches
5. **Automatically trigger the pipeline**

### 4. Monitor

Open Airflow at [http://localhost:8081](http://localhost:8081) (admin / admin) and watch all 11 tasks turn green.

**Expected runtime: ~5–8 minutes** (first run downloads NER model; subsequent runs ~3–5 min).

### 5. Stop

```bash
docker compose down        # Stop containers
docker compose down -v     # Stop + remove volumes
```

---

## Access Points

| Service | URL | Credentials |
|---|---|---|
| **Airflow Web UI** | [http://localhost:8081](http://localhost:8081) | admin / admin |
| **Spark Master UI** | [http://localhost:8080](http://localhost:8080) | — |
| **JupyterLab** | [http://localhost:8888](http://localhost:8888) | token: `governance` |
| **PostgreSQL** | localhost:5432 | airflow / airflow |

---

## Pipeline Output

After a successful run, the following data is produced:

### Data Volumes

| Layer | Table | Approximate Rows |
|---|---|---|
| Bronze | orders, customers, products, reviews, order_items | 500K, 103K, 10K, 200K, 500K |
| Silver | orders, customers, reviews | ~490K, 103K, 200K |
| Gold | revenue_aggregates, customer_rfm, customer_clv, churn_features, golden_customers | ~13K, 102K, 102K, 100K, ~100K |
| Quarantine | orders | ~10K |
| Streaming | clickstream | 20K |

### Governance Metrics

| Metric | Typical Value |
|---|---|
| DQ Score | ~92–93% |
| Adaptive Threshold | ~85% (learned) |
| Decision | PASS |
| Z-Score Anomalies | ~2,500 (0.5%) |
| IQR Anomalies | ~25,000 (5%) |
| Isolation Forest Anomalies | ~24,500 (5%) |
| Identity Resolution | 103K → ~100K (2,600+ duplicates resolved) |
| PII Post-Masking | 0 remaining |
| Contract Enforcement | ~435K valid, ~55K quarantined |

### Reports

Governance reports are saved as timestamped JSON files in:
```
data/metrics/governance_reports/silver_orders_YYYYMMDD_HHMMSS.json
```

Each report contains the full evaluation output from all 12 AI models.

---

## Configuration

Central configuration is in `config/config.yaml`. Key sections:

| Section | Purpose |
|---|---|
| `spark` | Spark session settings |
| `storage` | Medallion layer paths |
| `data_quality` | DQ thresholds, quarantine settings |
| `pii_detection` | NER model, confidence threshold, entity types |
| `data_contracts` | Contract directory, SLA enforcement |
| `identity_resolution` | Match thresholds, blocking keys |
| `streaming` | Landing directory, trigger interval |

---

## Key Modules

### Adaptive Governance Engine (`src/governance/adaptive_governance_engine.py`)
The central "brain" that orchestrates all AI components. A single `evaluate()` call runs 10 steps:
1. Compute DQ metrics with learned dimension weights
2. Z-score + IQR + Isolation Forest anomaly detection
3. Compute adaptive threshold from rolling history
4. Early warning trend analysis
5. Batch-level anomaly detection
6. Record run for future learning
7. PII drift check
8. PII threshold re-tuning
9. Regression-based weight learning
10. Final pass/fail/warn decision

### Anomaly Detector (`src/quality/anomaly_detector.py`)
Three complementary detection methods:
- **Z-Score**: Parametric, assumes normal distribution
- **IQR Fences**: Non-parametric, robust to skew
- **Isolation Forest**: ML-based, catches multi-dimensional anomalies

### Adaptive Scorer (`src/quality/adaptive_scorer.py`)
Self-tuning DQ scoring with:
- Rolling baseline threshold (μ − kσ)
- Inverse-mean dimension weight learning
- Linear regression weight learning (sklearn)
- Early warning system (consecutive decline detection)

### PII Detector (`src/pii_detection/pii_detector.py`)
Dual-mode PII detection:
- **Regex**: 8 Indian PII patterns (Aadhaar, PAN, phone, email, credit card, IPv4, address, IFSC)
- **NER**: Hugging Face `dslim/bert-base-NER` with chunked processing (450-char windows)

### Adaptive PII Tuner (`src/pii_detection/adaptive_pii_tuner.py`)
Feedback-driven PII threshold optimisation:
- Records detection feedback (TP/FP/FN/TN)
- Grid search over 50 candidate thresholds per entity type
- Maximises F1 score
- Detects PII-type drift via FN rate monitoring

### Identity Resolution (`src/governance/identity_resolution.py`)
Customer deduplication:
- Exact match on email + phone
- Jaro-Winkler fuzzy matching with configurable threshold
- Golden record creation (most recent canonical profile)

---

## License

© 2026 Kartikay Raniwala & Shreenam Tiwari. All rights reserved.
This project is submitted as part of a dissertation and may not be reproduced without permission.
