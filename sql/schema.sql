-- ============================================================
-- SENTINELPAY DATABASE SCHEMA
-- AI-Powered Real-Time Fraud & Anomaly Detection Platform
-- ============================================================


-- ============================================================
-- 1. TRANSACTIONS
-- ============================================================

CREATE TABLE transactions (
    transaction_id VARCHAR(20) PRIMARY KEY,
    transaction_time DOUBLE PRECISION NOT NULL,

    v1 DOUBLE PRECISION,
    v2 DOUBLE PRECISION,
    v3 DOUBLE PRECISION,
    v4 DOUBLE PRECISION,
    v5 DOUBLE PRECISION,
    v6 DOUBLE PRECISION,
    v7 DOUBLE PRECISION,
    v8 DOUBLE PRECISION,
    v9 DOUBLE PRECISION,
    v10 DOUBLE PRECISION,
    v11 DOUBLE PRECISION,
    v12 DOUBLE PRECISION,
    v13 DOUBLE PRECISION,
    v14 DOUBLE PRECISION,
    v15 DOUBLE PRECISION,
    v16 DOUBLE PRECISION,
    v17 DOUBLE PRECISION,
    v18 DOUBLE PRECISION,
    v19 DOUBLE PRECISION,
    v20 DOUBLE PRECISION,
    v21 DOUBLE PRECISION,
    v22 DOUBLE PRECISION,
    v23 DOUBLE PRECISION,
    v24 DOUBLE PRECISION,
    v25 DOUBLE PRECISION,
    v26 DOUBLE PRECISION,
    v27 DOUBLE PRECISION,
    v28 DOUBLE PRECISION,

    amount DOUBLE PRECISION,
    amount_log DOUBLE PRECISION,
    amount_zscore DOUBLE PRECISION,

    transaction_frequency_24h INTEGER,
    avg_transaction_amount DOUBLE PRECISION,
    amount_deviation DOUBLE PRECISION,

    pca_mean DOUBLE PRECISION,
    pca_std DOUBLE PRECISION,
    pca_abs_mean DOUBLE PRECISION,

    is_fraud INTEGER NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- 2. TRANSACTION INDEXES
-- ============================================================

CREATE INDEX idx_transactions_time
ON transactions(transaction_time);

CREATE INDEX idx_transactions_fraud
ON transactions(is_fraud);

CREATE INDEX idx_transactions_amount
ON transactions(amount);


-- ============================================================
-- 3. PREDICTIONS
-- ============================================================

CREATE TABLE predictions (
    prediction_id BIGSERIAL PRIMARY KEY,

    transaction_id VARCHAR(20)
        REFERENCES transactions(transaction_id),

    fraud_probability DOUBLE PRECISION,
    anomaly_score DOUBLE PRECISION,
    final_risk_score DOUBLE PRECISION,

    prediction VARCHAR(30),
    model_version VARCHAR(50),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE INDEX idx_predictions_transaction
ON predictions(transaction_id);


-- ============================================================
-- 4. ALERTS
-- ============================================================

CREATE TABLE alerts (
    alert_id BIGSERIAL PRIMARY KEY,

    transaction_id VARCHAR(20)
        REFERENCES transactions(transaction_id),

    risk_score DOUBLE PRECISION,
    alert_level VARCHAR(30),
    alert_reason TEXT,

    status VARCHAR(30) DEFAULT 'OPEN',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE INDEX idx_alerts_status
ON alerts(status);