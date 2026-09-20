# SentinelPay

AI-Powered Real-Time Fraud and Anomaly Detection Platform

SentinelPay is an end-to-end transaction risk intelligence system designed to detect potentially fraudulent and anomalous financial transactions using supervised machine learning, unsupervised anomaly detection, behavioral features, PostgreSQL, FastAPI, and an interactive Streamlit dashboard.

The system combines multiple detection signals into a unified transaction-level risk assessment rather than relying on a single fraud classifier.

---

## 1. Project Overview

Financial fraud detection is challenging because fraudulent transactions are rare, transaction behavior changes over time, and suspicious activity may not always match previously observed fraud patterns.

SentinelPay addresses this problem through a hybrid detection architecture that combines:

- Supervised fraud classification
- Unsupervised anomaly detection
- Transaction-level feature engineering
- Behavioral risk signals
- PostgreSQL-based transaction storage
- FastAPI inference services
- Streamlit-based monitoring and analysis

The final output is a transaction-level risk assessment containing fraud probability, anomaly score, final risk score, risk level, and detection reasons.

---

## 2. Problem Statement

Traditional fraud detection systems may depend heavily on historical fraud labels.

This creates two important challenges:

1. New fraud patterns may not have sufficient historical examples.
2. Unusual but previously unseen transaction behavior may be missed.

SentinelPay therefore treats fraud detection as a multi-signal risk intelligence problem.

The system evaluates both:

- "Does this transaction resemble previously known fraud?"
- "Does this transaction behave abnormally compared with normal transaction patterns?"

These signals are combined by a risk engine to produce a final transaction risk score.

---

## 3. Solution Architecture

```text
                         SENTINELPAY
                  TRANSACTION RISK INTELLIGENCE
                              |
                              v
                    +-------------------+
                    | Raw Transactions  |
                    +---------+---------+
                              |
                              v
                 +------------------------+
                 | Data Validation &      |
                 | Preprocessing          |
                 +-----------+------------+
                             |
                             v
                 +------------------------+
                 | Feature Engineering    |
                 |                        |
                 | Amount Features         |
                 | Behavioral Features    |
                 | Statistical Features  |
                 | PCA-derived Features  |
                 +-----------+------------+
                             |
                +------------+------------+
                |                         |
                v                         v
       +------------------+     +---------------------+
       | Supervised ML    |     | Anomaly Detection  |
       |                  |     |                     |
       | XGBoost          |     | PyTorch Autoencoder |
       | Random Forest    |     | Reconstruction      |
       +--------+---------+     +----------+----------+
                |                          |
                | Fraud Probability        | Anomaly Score
                +------------+-------------+
                             |
                             v
                    +------------------+
                    | Risk Engine      |
                    |                  |
                    | Signal Fusion    |
                    | Risk Calculation |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Final Risk Score |
                    +--------+---------+
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
        +----------------+      +------------------+
        | FastAPI        |      | PostgreSQL       |
        | Inference API  |      | Persistence      |
        +-------+--------+      +------------------+
                |
                v
        +----------------------+
        | Streamlit Dashboard  |
        |                      |
        | Transaction Analysis |
        | Risk Analysis        |
        | Model Information    |
        +----------------------+
```

---

## 4. Hybrid Detection Strategy

SentinelPay uses a hybrid supervised and unsupervised architecture.

### Supervised Fraud Detection

The supervised layer uses trained classification models to estimate the probability that a transaction belongs to the fraudulent class.

Models included in the project:

- XGBoost
- Random Forest

### Unsupervised Anomaly Detection

A PyTorch Autoencoder is used to identify unusual transaction behavior.

The Autoencoder learns a representation of transaction patterns and evaluates reconstruction error.

A higher reconstruction error can indicate that a transaction differs from learned normal behavior.

### Risk Fusion

The outputs of the fraud classifier and anomaly detector are passed to the risk engine.

```text
                 Transaction
                      |
          +-----------+-----------+
          |                       |
          v                       v
   Fraud Classifier        Autoencoder
          |                       |
          v                       v
 Fraud Probability          Anomaly Score
          |                       |
          +-----------+-----------+
                      |
                      v
                Risk Engine
                      |
                      v
              Final Risk Score
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
        LOW        MEDIUM       HIGH
```

This architecture allows SentinelPay to consider both known fraud patterns and unusual transaction behavior.

---

## 5. Key Features

### Real-Time Transaction Analysis

Users can enter transaction-level features through the Streamlit interface and request an immediate risk assessment.

The system returns:

- Risk score
- Risk level
- Fraud probability
- Anomaly probability
- Reconstruction error
- Detection reasons

### Fraud Detection

The supervised machine learning layer evaluates transaction characteristics against learned fraud patterns.

### Anomaly Detection

The Autoencoder evaluates whether transaction behavior is significantly different from learned patterns.

### Behavioral Risk Analysis

The system incorporates transaction behavior such as:

- Transaction frequency
- Average transaction amount
- Amount deviation
- Transaction amount statistics
- PCA-derived statistical features

### PostgreSQL Integration

PostgreSQL stores:

- Transactions
- Predictions
- Alerts

This provides persistent storage for transaction analysis and model outputs.

### FastAPI Inference Layer

FastAPI exposes the model inference functionality through an API.

This separates the model-serving layer from the user interface and makes the detection pipeline consumable by other applications.

### Streamlit Dashboard

The dashboard provides:

- System overview
- Transaction analysis
- Risk analysis
- Model information
- API status
- Detection results

---

## 6. Database Architecture

SentinelPay uses PostgreSQL for structured storage.

### Transactions

Stores transaction-level input and engineered features.

Important fields include:

- transaction_id
- transaction_time
- amount
- amount_log
- amount_zscore
- transaction_frequency_24h
- avg_transaction_amount
- amount_deviation
- PCA-derived features
- is_fraud

### Predictions

Stores model inference results.

Important fields include:

- transaction_id
- fraud_probability
- anomaly_score
- final_risk_score
- prediction
- model_version
- created_at

### Alerts

Stores risk alerts generated by the system.

Important fields include:

- transaction_id
- risk_score
- alert_level
- alert_reason
- status
- created_at

Database relationships:

```text
                 +----------------+
                 |  transactions  |
                 +-------+--------+
                         |
              +----------+----------+
              |                     |
              v                     v
      +---------------+     +---------------+
      | predictions   |     |    alerts     |
      +---------------+     +---------------+
```

The SQL implementation is available in:

```text
sql/schema.sql
```

Analytical SQL queries are maintained separately in:

```text
sql/analysis_queries.sql
```

---

## 7. Technology Stack

| Layer | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Deep Learning | PyTorch |
| Backend API | FastAPI |
| Database | PostgreSQL |
| Database Interface | pgAdmin |
| Dashboard | Streamlit |
| Visualization | Matplotlib |
| Version Control | Git |
| Repository | GitHub |

---

## 8. Project Structure

```text
SentinelPay/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   │
│   └── processed/
│       └── transactions_clean.csv
│
├── models/
│   ├── autoencoder_config.json
│   ├── autoencoder_scaler.pkl
│   ├── feature_columns.json
│   ├── feature_scaler.pkl
│   ├── fraud_autoencoder.pth
│   ├── model_metrics.json
│   ├── random_forest.pkl
│   └── xgboost_fraud_model.pkl
│
├── scripts/
│
├── sql/
│   ├── schema.sql
│   └── analysis_queries.sql
│
├── src/
│   ├── api.py
│   ├── inspect_data.py
│   ├── load_to_postgres.py
│   ├── preprocess_data.py
│   ├── risk_engine.py
│   ├── train_autoencoder.py
│   └── train_models.py
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 9. Dataset

The project uses transaction data containing numerical transaction features and fraud labels.

The processed database currently contains:

- Total transactions: 283,726
- Fraud cases: 473
- Fraud rate: approximately 0.1667%

The dataset is highly imbalanced, which reflects an important characteristic of real-world fraud detection systems.

Because fraudulent transactions represent a small fraction of total transactions, accuracy alone is not sufficient for evaluating the usefulness of a fraud detection system.

---

## 10. Machine Learning Pipeline

The machine learning workflow follows the sequence:

```text
Raw Dataset
     |
     v
Data Inspection
     |
     v
Data Cleaning
     |
     v
Feature Engineering
     |
     v
Feature Scaling
     |
     +-----------------------+
     |                       |
     v                       v
Fraud Classification    Autoencoder
     |                       |
     v                       v
Fraud Probability       Anomaly Score
     |                       |
     +-----------+-----------+
                 |
                 v
             Risk Engine
                 |
                 v
          Final Risk Score
```

The project separates preprocessing, model training, anomaly detection, and risk calculation into different Python modules.

---

## 11. Risk Engine

The risk engine acts as the decision layer between the machine learning models and the application.

Conceptually:

```text
Fraud Signal
     +
Anomaly Signal
     +
Behavioral Signals
     |
     v
Risk Engine
     |
     v
Final Risk Score
     |
     +--------+---------+
              |
        Risk Classification
              |
       +------+------+
       |      |      |
      LOW   MEDIUM   HIGH
```

This design allows additional risk signals to be incorporated without redesigning the complete machine learning pipeline.

---

## 12. FastAPI Service

The FastAPI application provides the model inference layer.

The API receives transaction features and returns the corresponding risk assessment.

Example API architecture:

```text
Client / Dashboard
        |
        v
     FastAPI
        |
        v
  Feature Processing
        |
        +----------------+
        |                |
        v                v
 Fraud Model       Autoencoder
        |                |
        +-------+--------+
                |
                v
            Risk Engine
                |
                v
          JSON Response
```

The FastAPI application is located at:

```text
src/api.py
```

---

## 13. Streamlit Dashboard

The Streamlit application provides the user-facing interface.

Main sections include:

### Dashboard

Provides an overview of the transaction monitoring system.

### Transaction Analysis

Allows users to enter transaction information and request risk analysis.

### Risk Analysis

Displays:

- Risk score
- Risk level
- Fraud probability
- Anomaly probability
- Reconstruction error
- Detection reasons

### Model Information

Displays the models and architecture used by SentinelPay.

The dashboard is implemented in:

```text
dashboard/app.py
```

---

## 14. Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd SentinelPay
```

Create and activate a Python virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 15. Database Configuration

Create a PostgreSQL database for SentinelPay.

The database schema is provided in:

```text
sql/schema.sql
```

Run the schema using pgAdmin or the PostgreSQL Query Tool.

The schema creates the required tables for:

- Transactions
- Predictions
- Alerts

Do not commit database passwords or private credentials to GitHub.

Use the provided environment template:

```text
.env.example
```

Create a local `.env` file containing your own database configuration.

Example:

```text
DATABASE_URL=your_database_connection_string
```

The actual `.env` file should remain private and should not be uploaded to GitHub.

---

## 16. Running the Project

### Start the FastAPI service

From the project root:

```bash
uvicorn src.api:app --reload
```

The FastAPI service will run locally.

The interactive API documentation can be accessed through the FastAPI Swagger interface.

### Start the Streamlit dashboard

Open a second terminal and activate the virtual environment.

Then run:

```bash
streamlit run dashboard/app.py
```

The Streamlit dashboard will open in the browser.

Both services should be running when the dashboard is configured to communicate with the FastAPI inference endpoint.

---

## 17. Configuration

Environment-specific configuration should be stored in `.env`.

A template is provided through:

```text
.env.example
```

Sensitive values such as:

- Database passwords
- API keys
- Private credentials
- Local secrets

should never be committed to the repository.

---

## 18. SQL Analytics

The repository contains a separate SQL analysis file:

```text
sql/analysis_queries.sql
```

It contains analytical queries for examining:

- Total transaction volume
- Fraud transaction count
- Fraud percentage
- Transaction samples
- Fraud distribution

This keeps database analysis reproducible and separates SQL analytics from the application code.

---

## 19. Model Information

SentinelPay currently integrates three primary models:

```text
1. XGBoost
2. Random Forest
3. PyTorch Autoencoder
```

The architecture can therefore be viewed as:

```text
             SentinelPay Models
                    |
        +-----------+-----------+
        |                       |
        v                       v
 Supervised Models       Unsupervised Model
        |                       |
   +----+----+                  |
   |         |                  |
   v         v                  v
XGBoost  Random Forest     Autoencoder
   |         |                  |
   +----+----+------------------+
             |
             v
        Risk Engine
```

---

## 20. Design Principles

The project follows several engineering principles:

### Modularity

Data processing, model training, inference, risk calculation, SQL, and dashboard components are separated.

### Reproducibility

Dependencies, SQL schemas, preprocessing scripts, and model artifacts are maintained inside the project structure.

### Separation of Concerns

The dashboard is separated from the FastAPI inference layer.

### Persistence

Transaction and prediction information can be stored in PostgreSQL rather than remaining only in application memory.

### Extensibility

Additional fraud models, anomaly detectors, behavioral signals, and alerting mechanisms can be incorporated into the architecture.

---

## 21. Why SentinelPay?

SentinelPay is designed as more than a standalone machine learning classifier.

The project connects:

```text
Machine Learning
       +
Anomaly Detection
       +
Feature Engineering
       +
Risk Engineering
       +
PostgreSQL
       +
FastAPI
       +
Streamlit
```

This creates an end-to-end transaction intelligence workflow covering data processing, model inference, risk calculation, persistence, API serving, and visualization.

---

## 22. Future Enhancements

Potential extensions include:

- Real-time transaction streaming
- Kafka-based event ingestion
- Redis-based low-latency feature storage
- Model monitoring
- Data drift detection
- Automated alert notifications
- SHAP-based model explainability
- Role-based dashboard access
- Historical customer behavior profiling
- Continuous model retraining
- Containerized deployment using Docker
- Cloud deployment
- API authentication and rate limiting

---

## 23. Project Status

Current implementation includes:

- Transaction preprocessing
- Feature engineering
- Supervised fraud models
- Autoencoder-based anomaly detection
- Risk engine
- PostgreSQL schema
- FastAPI inference service
- Streamlit dashboard
- SQL analytics
- Model artifacts
- Environment configuration
- Reproducible dependency specification

---

## 24. Repository Purpose

This repository demonstrates the design and implementation of an end-to-end AI-based fraud and anomaly detection platform.

The emphasis is on integrating machine learning with software engineering, database systems, backend APIs, and an operational dashboard rather than presenting a machine learning model in isolation.

---

## Author

Shubhechchha Hazra
National Institute of Technology, Raipur

AI/ML | Data Science | Software Engineering

Connect
GitHub: https://github.com/shubhechchha-23
LinkedIn: https://www.linkedin.com/in/shubhechchha232
Email: shubhechchha232@gmail.com
