from pathlib import Path
import sys
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ------------------------------------------------------------
# Add src directory to Python path
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


# ------------------------------------------------------------
# Import SentinelPay Risk Engine
# ------------------------------------------------------------

from risk_engine import SentinelPayRiskEngine


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SentinelPay Fraud Detection API",
    description=(
        "Real-time AI-powered transaction fraud "
        "and anomaly detection API."
    ),
    version="1.0.0"
)


# ============================================================
# LOAD RISK ENGINE ONCE
# ============================================================

try:

    risk_engine = SentinelPayRiskEngine()

    MODEL_STATUS = "loaded"

except Exception as e:

    risk_engine = None

    MODEL_STATUS = f"failed: {str(e)}"


# ============================================================
# TRANSACTION INPUT MODEL
# ============================================================

class TransactionRequest(BaseModel):

    Time: float = Field(
        default=0,
        description="Transaction timestamp"
    )

    Amount: float = Field(
        default=0,
        ge=0,
        description="Transaction amount"
    )

    V1: float = 0
    V2: float = 0
    V3: float = 0
    V4: float = 0
    V5: float = 0
    V6: float = 0
    V7: float = 0
    V8: float = 0
    V9: float = 0
    V10: float = 0
    V11: float = 0
    V12: float = 0
    V13: float = 0
    V14: float = 0
    V15: float = 0
    V16: float = 0
    V17: float = 0
    V18: float = 0
    V19: float = 0
    V20: float = 0
    V21: float = 0
    V22: float = 0
    V23: float = 0
    V24: float = 0
    V25: float = 0
    V26: float = 0
    V27: float = 0
    V28: float = 0

    transaction_frequency_24h: float = 0
    avg_transaction_amount: float = 0
    amount_deviation: float = 0
    amount_zscore: float = 0

    pca_mean: float = 0
    pca_std: float = 0
    pca_abs_mean: float = 0


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {

        "service": "SentinelPay",

        "description":
            "AI-powered real-time fraud detection system",

        "status": "running",

        "model_status":
            MODEL_STATUS

    }


# ============================================================
# SYSTEM HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "risk_engine":
            "available"
            if risk_engine is not None
            else "unavailable"

    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return {

        "system":
            "SentinelPay AI Fraud Detection",

        "models": [

            "XGBoost",

            "Random Forest",

            "PyTorch Autoencoder"

        ],

        "architecture":
            "Hybrid supervised + unsupervised anomaly detection",

        "output":
            "Real-time transaction risk score",

        "risk_scale":
            "0-100"

    }


# ============================================================
# REAL-TIME PREDICTION
# ============================================================

@app.post("/predict")
def predict_transaction(
    transaction: TransactionRequest
):

    if risk_engine is None:

        raise HTTPException(
            status_code=503,
            detail="Risk engine is not available."
        )


    try:

        # Convert Pydantic object to dictionary

        transaction_data = (
            transaction.model_dump()
        )


        # ----------------------------------------------------
        # Rename fields for Risk Engine
        # ----------------------------------------------------

        transaction_data[
            "transaction_time"
        ] = transaction_data.pop(
            "Time"
        )


        transaction_data[
            "amount"
        ] = transaction_data.pop(
            "Amount"
        )


        # ----------------------------------------------------
        # Rename PCA variables
        # ----------------------------------------------------

        for i in range(1, 29):

            old_name = f"V{i}"

            new_name = f"v{i}"

            if old_name in transaction_data:

                transaction_data[
                    new_name
                ] = transaction_data.pop(
                    old_name
                )


        # ----------------------------------------------------
        # Run AI risk engine
        # ----------------------------------------------------

        result = risk_engine.predict(
            transaction_data
        )


        # ----------------------------------------------------
        # Return API response
        # ----------------------------------------------------

        return {

            "success": True,

            "transaction": {

                "amount":
                    transaction.Amount,

                "timestamp":
                    transaction.Time

            },

            "risk_analysis": result

        }


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "api:app",

        host="127.0.0.1",

        port=8000,

        reload=True

    )