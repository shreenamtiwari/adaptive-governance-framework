# Adaptive Data Governance Framework for E-Commerce

An intelligent, policy-driven data governance framework built with **PySpark**, **Delta Lake**, and **Apache Airflow** — designed for e-commerce data platforms that demand robust data quality, PII protection, and lineage tracking at scale.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This framework provides an end-to-end adaptive data governance solution for e-commerce platforms. It implements a **Medallion Architecture** (Bronze → Silver → Gold) with built-in:

- **Data Quality Enforcement** — Schema validation, anomaly detection, and quarantine workflows
- **PII Detection & Masking** — Automated scanning and tokenization of personally identifiable information
- **Policy-as-Code Governance** — Declarative governance rules that adapt based on data profiling results
- **Data Lineage Tracking** — Full traceability from raw ingestion through to curated gold-layer datasets
- **Orchestration** — Apache Airflow DAGs for scheduling and monitoring governance pipelines

### Key Technologies

| Component          | Technology                  |
|--------------------|-----------------------------| 
| Processing Engine  | Apache PySpark              |
| Storage Layer      | Delta Lake                  |
| Orchestration      | Apache Airflow              |
| Cloud Platform     | Google Cloud Platform (GCP) |
| Language           | Python 3.10+                |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Apache Airflow (Orchestration)               │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│   │ Ingest   │───▶│ Quality  │───▶│ PII      │───▶│ Govern   │    │
│   │ DAG      │    │ DAG      │    │ Scan DAG │    │ DAG      │    │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘    │
└────────┬───────────────┬───────────────┬───────────────┬───────────┘
         │               │               │               │
         ▼               ▼               ▼               ▼
┌────────────┐   ┌──────────────┐  ┌───────────┐  ┌────────────────┐
│  Raw Data  │   │ Bronze Layer │  │  Silver   │  │   Gold Layer   │
│  (Landing) │──▶│ (Delta Lake) │─▶│  Layer    │─▶│  (Delta Lake)  │
└────────────┘   └──────────────┘  └─────┬─────┘  └────────────────┘
                                         │
                                   ┌─────▼─────┐
                                   │ Quarantine │
                                   │  (Failed)  │
                                   └───────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     Cross-Cutting Concerns                         │
│  ┌────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │  Lineage   │  │  PII         │  │  Adaptive Policy Engine   │  │
│  │  Tracking  │  │  Detection   │  │  (Policy-as-Code)         │  │
│  └────────────┘  └──────────────┘  └───────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

> **TODO:** Replace with a detailed architecture diagram (e.g., draw.io, Lucidchart, or Mermaid export).

---

## Project Structure

```
adaptive-governance-framework/
├── .gitignore                  # Ignore rules for Python, PySpark, data, Airflow, secrets
├── .env.example                # Environment variable template
├── README.md                   # This file
├── data/
│   ├── bronze/                 # Raw ingested data (append-only Delta tables)
│   ├── silver/                 # Cleaned, validated, and deduplicated data
│   ├── gold/                   # Business-level aggregations and curated datasets
│   ├── raw/                    # Landing zone for source system extracts
│   └── quarantine/             # Records that failed quality or governance checks
├── src/
│   ├── ingestion/              # Data ingestion modules (batch & streaming)
│   ├── quality/                # Data quality rules, validators, and profilers
│   ├── governance/             # Policy engine, lineage, and access control
│   ├── pii_detection/          # PII scanning, classification, and masking
│   └── utils/                  # Shared utilities, logging, and Spark session helpers
├── tests/                      # Unit and integration tests
├── airflow/
│   ├── dags/                   # Airflow DAG definitions
│   ├── logs/                   # Airflow execution logs
│   └── plugins/                # Custom Airflow operators and hooks
├── config/                     # YAML/JSON configuration files for policies and schemas
├── docs/                       # Documentation, ADRs, and design specs
├── models/                     # Trained ML models (e.g., PII classifiers)
└── notebooks/                  # Jupyter/Databricks notebooks for exploration
```

---

## Prerequisites

- **Python** 3.10 or higher
- **Apache Spark** 3.4+ with PySpark
- **Delta Lake** 2.4+
- **Apache Airflow** 2.7+
- **Google Cloud SDK** (for GCP integration)
- **Docker & Docker Compose** (recommended for local Airflow)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/KartikayRaniwala/adaptive-governance-framework.git
cd adaptive-governance-framework
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your actual credentials and configuration
```

### 5. Initialize Airflow

```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### 6. Run Tests

```bash
pytest tests/ -v --tb=short
```

### 7. Start the Framework

```bash
# Start Airflow (in separate terminals)
airflow webserver --port 8080
airflow scheduler

# Or use Docker Compose
docker-compose up -d
```

---

## Usage

### Running a Governance Pipeline

```python
from src.ingestion.batch_ingester import BatchIngester
from src.quality.validator import DataQualityValidator
from src.pii_detection.scanner import PIIScanner
from src.governance.policy_engine import PolicyEngine

# Initialize components
ingester = BatchIngester(source="gcs://ecommerce-raw/orders/")
validator = DataQualityValidator(config="config/quality_rules.yaml")
scanner = PIIScanner(config="config/pii_policies.yaml")
engine = PolicyEngine(config="config/governance_policies.yaml")

# Execute pipeline
raw_df = ingester.ingest()
validated_df, quarantined_df = validator.validate(raw_df)
masked_df = scanner.scan_and_mask(validated_df)
engine.apply_policies(masked_df)
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.