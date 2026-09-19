-- ============================================================
-- SENTINELPAY ANALYSIS QUERIES
-- PostgreSQL Analytics & Fraud Monitoring
-- ============================================================


-- ============================================================
-- 1. TOTAL TRANSACTIONS
-- ============================================================

SELECT COUNT(*) AS total_transactions
FROM transactions;


-- ============================================================
-- 2. TOTAL FRAUDULENT TRANSACTIONS
-- ============================================================

SELECT COUNT(*) AS fraud_transactions
FROM transactions
WHERE is_fraud = 1;


-- ============================================================
-- 3. FRAUD DISTRIBUTION
-- ============================================================

SELECT
    is_fraud,
    COUNT(*) AS transaction_count,
    ROUND(
        COUNT(*) * 100.0 /
        SUM(COUNT(*)) OVER (),
        4
    ) AS percentage
FROM transactions
GROUP BY is_fraud
ORDER BY is_fraud;


-- ============================================================
-- 4. TRANSACTION STATISTICS
-- ============================================================

SELECT
    COUNT(*) AS total_transactions,
    ROUND(AVG(amount)::numeric, 2) AS average_amount,
    ROUND(MAX(amount)::numeric, 2) AS maximum_amount,
    ROUND(MIN(amount)::numeric, 2) AS minimum_amount
FROM transactions;


-- ============================================================
-- 5. FRAUD TRANSACTION STATISTICS
-- ============================================================

SELECT
    COUNT(*) AS fraud_count,
    ROUND(AVG(amount)::numeric, 2) AS average_fraud_amount,
    ROUND(MAX(amount)::numeric, 2) AS maximum_fraud_amount
FROM transactions
WHERE is_fraud = 1;


-- ============================================================
-- 6. TOP HIGH-VALUE TRANSACTIONS
-- ============================================================

SELECT
    transaction_id,
    transaction_time,
    amount,
    amount_zscore,
    is_fraud
FROM transactions
ORDER BY amount DESC
LIMIT 20;


-- ============================================================
-- 7. HIGH-RISK TRANSACTIONS BY AMOUNT Z-SCORE
-- ============================================================

SELECT
    transaction_id,
    amount,
    amount_zscore,
    transaction_frequency_24h,
    amount_deviation,
    is_fraud
FROM transactions
ORDER BY ABS(amount_zscore) DESC
LIMIT 20;


-- ============================================================
-- 8. TRANSACTIONS WITH HIGH ACTIVITY
-- ============================================================

SELECT
    transaction_id,
    amount,
    transaction_frequency_24h,
    avg_transaction_amount,
    amount_deviation,
    is_fraud
FROM transactions
ORDER BY transaction_frequency_24h DESC
LIMIT 20;


-- ============================================================
-- 9. MODEL PREDICTION SUMMARY
-- ============================================================

SELECT
    prediction,
    COUNT(*) AS prediction_count,
    ROUND(AVG(fraud_probability)::numeric, 4) AS avg_fraud_probability,
    ROUND(AVG(final_risk_score)::numeric, 4) AS avg_risk_score
FROM predictions
GROUP BY prediction
ORDER BY avg_risk_score DESC;


-- ============================================================
-- 10. HIGH-RISK PREDICTIONS
-- ============================================================

SELECT
    transaction_id,
    fraud_probability,
    anomaly_score,
    final_risk_score,
    prediction,
    model_version,
    created_at
FROM predictions
WHERE final_risk_score >= 0.80
ORDER BY final_risk_score DESC;


-- ============================================================
-- 11. ALERT SUMMARY
-- ============================================================

SELECT
    alert_level,
    status,
    COUNT(*) AS alert_count
FROM alerts
GROUP BY alert_level, status
ORDER BY alert_count DESC;


-- ============================================================
-- 12. OPEN ALERTS
-- ============================================================

SELECT
    alert_id,
    transaction_id,
    risk_score,
    alert_level,
    alert_reason,
    created_at
FROM alerts
WHERE status = 'OPEN'
ORDER BY risk_score DESC;


-- ============================================================
-- 13. FRAUD VS NON-FRAUD AMOUNT COMPARISON
-- ============================================================

SELECT
    is_fraud,
    ROUND(AVG(amount)::numeric, 2) AS average_amount,
    ROUND(AVG(amount_zscore)::numeric, 4) AS average_amount_zscore,
    ROUND(AVG(amount_deviation)::numeric, 4) AS average_amount_deviation
FROM transactions
GROUP BY is_fraud
ORDER BY is_fraud;