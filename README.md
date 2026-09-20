# SentinelPay

### AI-Powered Real-Time Fraud & Anomaly Detection Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PostgreSQL-Database-4169E1?style=for-the-badge&logo=postgresql&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-FF6600?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Git-GitHub-F05032?style=for-the-badge&logo=git&logoColor=white"/>
</p>

<p align="center">
  <b>Detect → Analyze → Score → Explain → Alert</b>
</p>

<p align="center">
  An end-to-end fraud intelligence platform combining machine learning,
  anomaly detection, behavioral analytics, risk scoring,
  PostgreSQL persistence and API-driven monitoring.
</p>

---

## 🛡️ Overview

**SentinelPay** is an AI-powered fraud detection and risk intelligence platform designed to transform transaction data into actionable risk signals.

Instead of treating fraud detection as a simple:

```text
Transaction → Fraud / Not Fraud
```

SentinelPay follows a multi-layer intelligence pipeline:

```text
Transaction
     ↓
Data Processing
     ↓
Feature Engineering
     ↓
Supervised ML ─────────┐
                       ├──→ Risk Fusion Engine
Anomaly Detection ────┘
                       ↓
                Final Risk Score
                       ↓
             Prediction + Explanation
                       ↓
                 Alert Generation
                       ↓
          PostgreSQL + FastAPI + Dashboard
```

---

## 🎯 Why SentinelPay?

Financial fraud is rarely defined by a single signal.

A transaction can become suspicious because of a combination of:

* Unusual transaction amount
* Abnormal transaction frequency
* Amount deviation
* Behavioral patterns
* Machine-learning fraud probability
* Anomaly characteristics

SentinelPay combines these signals into a unified risk intelligence layer rather than depending exclusively on a single classification output.

---

# 🧠 System Architecture

```text
                         SENTINELPAY
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   Transaction Dataset   │
                 │      Raw / Processed    │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │    Data Processing      │
                 │                         │
                 │ Cleaning & Validation   │
                 │ Feature Engineering     │
                 └────────────┬────────────┘
                              │
                              ▼
             ┌──────────────────────────────────┐
             │         DETECTION LAYER          │
             │                                  │
             │  ┌────────────┐  ┌────────────┐ │
             │  │ Supervised │  │  Anomaly   │ │
             │  │    ML      │  │ Detection  │ │
             │  └─────┬──────┘  └──────┬─────┘ │
             └────────┼─────────────────┼───────┘
                      │                 │
                      └────────┬────────┘
                               ▼
                 ┌─────────────────────────┐
                 │       RISK ENGINE       │
                 │                         │
                 │ ML Probability          │
                 │ Anomaly Score           │
                 │ Behavioral Signals      │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     FINAL RISK SCORE    │
                 │                         │
                 │ LOW / MEDIUM / HIGH     │
                 │        / CRITICAL       │
                 └────────────┬────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
          ┌─────────────────┐   ┌─────────────────┐
          │   PostgreSQL    │   │     FastAPI     │
          │                 │   │                 │
          │ Transactions    │   │ REST Endpoints  │
          │ Predictions     │   │ Risk Analysis   │
          │ Alerts          │   │ Monitoring      │
          └────────┬────────┘   └────────┬────────┘
                   │                     │
                   └──────────┬──────────┘
                              ▼
                    ┌──────────────────┐
                    │    Streamlit     │
                    │  Risk Dashboard  │
                    └──────────────────┘
```

---

# 🔄 Detection Pipeline

## 01 — Data Processing

Raw transaction records are transformed into model-ready data.

The processing layer handles:

* Data cleaning
* Validation
* Transformation
* Feature preparation
* Dataset generation

```text
Raw Transaction Data
        ↓
Validation
        ↓
Cleaning
        ↓
Processed Dataset
```

---

## 02 — Feature Engineering

Transaction-level signals are transformed into analytical features.

| Feature                    | Purpose                      |
| -------------------------- | ---------------------------- |
| Transaction Amount         | Monetary behavior            |
| Log Amount                 | Distribution transformation  |
| Amount Z-Score             | Amount deviation             |
| Transaction Frequency      | Behavioral velocity          |
| Average Transaction Amount | Customer baseline            |
| PCA Features               | Feature-space representation |
| Fraud Label                | Supervised learning target   |

---

# 🤖 Machine Learning Layer

SentinelPay uses a hybrid detection architecture.

```text
                 Transaction Features
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
      Supervised ML            Anomaly Detection
            │                         │
            ▼                         ▼
     Fraud Probability         Anomaly Score
            │                         │
            └────────────┬────────────┘
                         ▼
                  Risk Engine
```

## Supervised Fraud Detection

The supervised layer learns patterns associated with known fraudulent transactions.

The project uses technologies including:

* Scikit-learn
* XGBoost
* PyTorch

The output is a fraud probability:

```text
P(Fraud | Transaction)
```

This probability becomes an input to the downstream risk engine.

---

## 🔎 Anomaly Detection

Known fraud patterns are not the only source of risk.

The anomaly layer identifies transaction behavior that deviates from expected patterns.

Examples include:

```text
Unusual Amount
      ↓
Abnormal Frequency
      ↓
Behavioral Deviation
      ↓
Feature-Space Irregularity
      ↓
Anomaly Score
```

This provides a complementary detection perspective alongside supervised classification.

---

# 🎯 Risk Fusion Engine

The risk engine acts as the central intelligence layer.

```text
              ML Probability
                    │
                    ▼
             ┌─────────────┐
             │             │
Anomaly ────►│ Risk Engine │◄──── Behavioral Signals
             │             │
             └──────┬──────┘
                    │
                    ▼
             Final Risk Score
```

The architecture separates three concepts:

### Prediction

What does the supervised model estimate?

### Anomaly

How unusual is the transaction?

### Risk

What combined risk signal is produced by the detection pipeline?

This separation makes the architecture easier to extend and monitor.

---

# 🚨 Alert Intelligence

High-risk transactions can generate structured alerts containing:

* Transaction ID
* Risk score
* Alert level
* Alert reason
* Timestamp
* Status

```text
Transaction
     ↓
Risk Score
     ↓
Risk Threshold
     ↓
Alert Generation
     ↓
PostgreSQL
     ↓
API / Dashboard
```

---

# 🗄️ PostgreSQL Data Layer

PostgreSQL acts as the persistent analytical backend.

The core database model contains three primary entities:

```text
                    transactions
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        predictions              alerts
```

### transactions

Stores transaction-level features and fraud labels.

### predictions

Stores:

* Fraud probability
* Anomaly score
* Final risk score
* Prediction
* Model version
* Timestamp

### alerts

Stores:

* Risk score
* Alert level
* Alert reason
* Status
* Timestamp

Indexes are maintained for frequently queried fields such as transaction ID, timestamp, fraud label, amount and alert status.

---

# 🧱 Database Schema

The PostgreSQL schema is maintained separately in:

```text
sql/schema.sql
```

This keeps the database architecture version-controlled, modular and reproducible.

---

# ⚡ FastAPI Layer

SentinelPay exposes its intelligence layer through FastAPI.

```text
PostgreSQL
     ↓
ML / Detection Layer
     ↓
Risk Engine
     ↓
FastAPI
     ↓
Dashboard / Client
```

The API layer is designed to make the detection engine easier to integrate with:

* Payment systems
* Financial applications
* Fraud investigation platforms
* Risk monitoring tools
* Internal analytics systems

FastAPI also provides interactive API documentation through Swagger UI.

```text
http://127.0.0.1:8000/docs
```

---

# 📊 Streamlit Dashboard

The Streamlit dashboard provides an interactive monitoring layer for exploring:

* Transaction activity
* Fraud patterns
* Risk scores
* Anomaly signals
* Alerts
* Model outputs

The objective is to translate model-level outputs into an interpretable monitoring experience.

---

# 🧰 Technology Stack

| Layer             | Technology           |
| ----------------- | -------------------- |
| Programming       | Python               |
| Data Processing   | Pandas, NumPy        |
| Machine Learning  | Scikit-learn         |
| Gradient Boosting | XGBoost              |
| Deep Learning     | PyTorch              |
| Database          | PostgreSQL           |
| Database Access   | SQLAlchemy, Psycopg2 |
| Backend API       | FastAPI              |
| API Server        | Uvicorn              |
| Dashboard         | Streamlit            |
| Configuration     | python-dotenv        |
| Version Control   | Git / GitHub         |

---

# 📁 Project Structure

```text
SentinelPay/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── trained_models/
│
├── sql/
│   └── schema.sql
│
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── anomaly_detection.py
│   ├── risk_engine.py
│   ├── load_to_postgres.py
│   └── api.py
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 🚀 Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/shubhechchha232-codecommitqueen/SentinelPay.git
cd SentinelPay
```

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Create a local `.env` file:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sentinelpay
DB_USER=postgres
DB_PASSWORD=YOUR_POSTGRES_PASSWORD
```

Never commit the real `.env` file to GitHub.

---

# 🗄️ PostgreSQL Setup

Create a PostgreSQL database named:

```text
sentinelpay
```

Then execute:

```text
sql/schema.sql
```

using pgAdmin or another PostgreSQL client.

The schema creates the required:

```text
Tables
Relationships
Indexes
Constraints
```

---

# ▶️ Running the Pipeline

### Data Processing

```bash
python src/data_processing.py
```

### Model Training

```bash
python src/train_model.py
```

### Anomaly Detection

```bash
python src/anomaly_detection.py
```

### Risk Engine

```bash
python src/risk_engine.py
```

### PostgreSQL Loader

```bash
python src/load_to_postgres.py
```

### Start FastAPI

```bash
uvicorn src.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

### Start Dashboard

```bash
streamlit run dashboard/app.py
```

---

# 📦 Data Management

Large raw and processed CSV datasets are intentionally excluded from the public repository because they exceed GitHub's standard per-file size limit.

Expected local structure:

```text
data/
├── raw/
│   └── creditcard.csv
│
└── processed/
    └── transactions_clean.csv
```

The datasets remain available locally for:

* Model training
* Data processing
* PostgreSQL loading
* Testing

The public repository therefore focuses on the reusable engineering layer:

```text
Source Code
SQL Schema
Configuration
Dependencies
Documentation
```

---

# 🧩 Engineering Principles

### Modular Architecture

Data processing, machine learning, anomaly detection, risk scoring, API services and visualization are separated into independent components.

### Reproducibility

Dependencies are maintained in `requirements.txt`, while environment configuration is externalized through `.env`.

### Persistent Intelligence

Transactions, predictions and alerts are stored independently from model execution.

### Extensibility

The architecture allows new models and detection strategies to be introduced without redesigning the complete application.

### Operational Orientation

The system is structured around:

```text
Detection
   ↓
Risk Scoring
   ↓
Persistence
   ↓
API
   ↓
Monitoring
   ↓
Alerts
```

---

# 💡 What Makes SentinelPay Different?

A conventional introductory fraud-detection project often follows:

```text
Dataset
   ↓
Train Model
   ↓
Predict
   ↓
Accuracy
```

SentinelPay expands the workflow into a broader engineering system:

```text
              Transaction Data
                     │
                     ▼
              Data Processing
                     │
                     ▼
             Feature Engineering
                     │
             ┌───────┴────────┐
             ▼                ▼
       Supervised ML    Anomaly Detection
             │                │
             └───────┬────────┘
                     ▼
                Risk Engine
                     │
                     ▼
                 Risk Score
                     │
              ┌──────┴──────┐
              ▼             ▼
         PostgreSQL       FastAPI
              │             │
              └──────┬──────┘
                     ▼
                 Dashboard
                     │
                     ▼
                   Alerts
```

The focus is therefore not only on model training, but on integrating machine learning into a complete:

```text
Data → ML → Backend → Database → Risk → Monitoring
```

workflow.

---

# ✨ Key Highlights

| Capability                  | SentinelPay |
| --------------------------- | ----------- |
| Supervised Machine Learning | ✓           |
| Anomaly Detection           | ✓           |
| Feature Engineering         | ✓           |
| Risk Fusion Engine          | ✓           |
| Alert Generation            | ✓           |
| PostgreSQL Persistence      | ✓           |
| FastAPI Backend             | ✓           |
| Streamlit Dashboard         | ✓           |
| Environment Configuration   | ✓           |
| Modular Architecture        | ✓           |
| Reproducible Setup          | ✓           |

---

# 🔮 Future Roadmap

The architecture can evolve toward real-time transaction intelligence.

```text
Real-Time Transaction
          ↓
Event Stream
          ↓
Online Feature Engineering
          ↓
Model Inference
          ↓
Dynamic Risk Score
          ↓
Automated Alerting
          ↓
Fraud Analyst Dashboard
```

Potential extensions include:

* Real-time transaction streaming
* Kafka integration
* Redis caching
* Online feature engineering
* Model monitoring
* Concept-drift detection
* SHAP-based explanations
* Feature-store integration
* Docker deployment
* CI/CD automation
* Cloud deployment
* Analyst feedback loops
* Continuous model improvement

---

# 👩‍💻 Author

## Shubhechchha Hazra

**B.Tech — Biomedical Engineering**
**National Institute of Technology, Raipur**

GitHub:
https://github.com/shubhechchha-23

LinkedIn:
https://www.linkedin.com/in/shubhechchha232

Email:
[shubhechchha232@gmail.com](mailto:shubhechchha232@gmail.com)
