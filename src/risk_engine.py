from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# SENTINELPAY - REAL-TIME RISK ENGINE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"


# ============================================================
# LOAD FEATURE CONFIGURATION
# ============================================================

with open(
    MODEL_DIR / "feature_columns.json",
    "r"
) as file:

    FEATURE_COLUMNS = json.load(file)


# ============================================================
# LOAD SUPERVISED MODELS
# ============================================================

xgb_model = joblib.load(
    MODEL_DIR / "xgboost_fraud_model.pkl"
)


rf_model = joblib.load(
    MODEL_DIR / "random_forest.pkl"
)


# ============================================================
# LOAD SCALER
# ============================================================

rf_scaler = joblib.load(
    MODEL_DIR / "feature_scaler.pkl"
)


# ============================================================
# AUTOENCODER
# ============================================================

class FraudAutoencoder(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                64
            ),

            nn.ReLU(),

            nn.BatchNorm1d(64),

            nn.Dropout(0.10),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                16
            )
        )

        self.decoder = nn.Sequential(

            nn.Linear(
                16,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                input_dim
            )
        )


    def forward(self, x):

        encoded = self.encoder(x)

        decoded = self.decoder(
            encoded
        )

        return decoded


# ============================================================
# LOAD AUTOENCODER
# ============================================================

autoencoder_checkpoint = torch.load(

    MODEL_DIR /
    "fraud_autoencoder.pth",

    map_location="cpu"
)


autoencoder = FraudAutoencoder(
    autoencoder_checkpoint[
        "input_dimension"
    ]
)


autoencoder.load_state_dict(
    autoencoder_checkpoint[
        "model_state_dict"
    ]
)


autoencoder.eval()


# ============================================================
# AUTOENCODER SCALER
# ============================================================

autoencoder_scaler = joblib.load(

    MODEL_DIR /
    "autoencoder_scaler.pkl"

)


# ============================================================
# ANOMALY THRESHOLD
# ============================================================

ANOMALY_THRESHOLD = (
    autoencoder_checkpoint[
        "anomaly_threshold"
    ]
)


# ============================================================
# RISK ENGINE
# ============================================================

class SentinelPayRiskEngine:

    """
    Real-time transaction risk scoring engine.

    Combines:

    1. XGBoost fraud probability
    2. Random Forest probability
    3. PyTorch reconstruction error
    """

    def __init__(self):

        self.xgb = xgb_model

        self.rf = rf_model

        self.rf_scaler = rf_scaler

        self.autoencoder = autoencoder

        self.autoencoder_scaler = (
            autoencoder_scaler
        )

        self.anomaly_threshold = (
            ANOMALY_THRESHOLD
        )


    # ========================================================
    # FEATURE PREPARATION
    # ========================================================

    def prepare_features(
        self,
        transaction
    ):

        row = {}

        for feature in FEATURE_COLUMNS:

            row[feature] = transaction.get(
                feature,
                0
            )


        dataframe = pd.DataFrame(
            [row]
        )


        dataframe = dataframe.replace(
            [np.inf, -np.inf],
            np.nan
        )


        dataframe = dataframe.fillna(0)


        return dataframe[
            FEATURE_COLUMNS
        ]


    # ========================================================
    # CALCULATE RISK
    # ========================================================

    def predict(
        self,
        transaction
    ):

        # ----------------------------------------------------
        # Prepare features
        # ----------------------------------------------------

        X = self.prepare_features(
            transaction
        )


        # ----------------------------------------------------
        # XGBoost
        # ----------------------------------------------------

        xgb_probability = float(

            self.xgb.predict_proba(
                X
            )[0][1]

        )


        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        X_scaled = (
            self.rf_scaler.transform(
                X
            )
        )


        rf_probability = float(

            self.rf.predict_proba(
                X_scaled
            )[0][1]

        )


        # ----------------------------------------------------
        # Autoencoder
        # ----------------------------------------------------

        X_auto = (
            self.autoencoder_scaler.transform(
                X
            )
        )


        X_tensor = torch.tensor(
            X_auto,
            dtype=torch.float32
        )


        with torch.no_grad():

            reconstruction = (
                self.autoencoder(
                    X_tensor
                )
            )


            reconstruction_error = float(

                torch.mean(
                    (
                        reconstruction
                        - X_tensor
                    ) ** 2
                ).item()

            )


        # ----------------------------------------------------
        # NORMALIZED ANOMALY SCORE
        # ----------------------------------------------------

        anomaly_score = (

            reconstruction_error
            /
            max(
                self.anomaly_threshold,
                1e-8
            )

        )


        anomaly_score = min(
            anomaly_score,
            5.0
        )


        anomaly_probability = min(

            anomaly_score / 5.0,

            1.0

        )


        # ----------------------------------------------------
        # ENSEMBLE RISK
        # ----------------------------------------------------

        combined_probability = (

            0.60 * xgb_probability

            +

            0.20 * rf_probability

            +

            0.20 * anomaly_probability

        )


        # ----------------------------------------------------
        # RISK SCORE
        # ----------------------------------------------------

        risk_score = round(

            combined_probability
            * 100,

            2

        )


        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        if risk_score >= 80:

            risk_level = "CRITICAL"

        elif risk_score >= 60:

            risk_level = "HIGH"

        elif risk_score >= 30:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"


        # ----------------------------------------------------
        # ALERT DECISION
        # ----------------------------------------------------

        if risk_score >= 60:

            alert = True

        else:

            alert = False


        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        reasons = []


        if xgb_probability >= 0.50:

            reasons.append(
                "High supervised fraud probability"
            )


        if rf_probability >= 0.50:

            reasons.append(
                "Random Forest indicates suspicious behavior"
            )


        if anomaly_probability >= 0.50:

            reasons.append(
                "Transaction differs significantly from normal behavior"
            )


        amount = transaction.get(
            "amount",
            0
        )


        amount_zscore = transaction.get(
            "amount_zscore",
            0
        )


        if abs(amount_zscore) >= 3:

            reasons.append(
                "Unusual transaction amount"
            )


        frequency = transaction.get(
            "transaction_frequency_24h",
            0
        )


        if frequency >= 10:

            reasons.append(
                "Unusually high transaction frequency"
            )


        if not reasons:

            reasons.append(
                "No major suspicious signal detected"
            )


        # ----------------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------------

        return {

            "risk_score":
                risk_score,

            "risk_level":
                risk_level,

            "fraud_probability":
                round(
                    xgb_probability * 100,
                    2
                ),

            "random_forest_probability":
                round(
                    rf_probability * 100,
                    2
                ),

            "anomaly_probability":
                round(
                    anomaly_probability * 100,
                    2
                ),

            "reconstruction_error":
                round(
                    reconstruction_error,
                    6
                ),

            "alert":
                alert,

            "reasons":
                reasons

        }


# ============================================================
# TEST THE ENGINE
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "       SENTINELPAY RISK ENGINE TEST"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Load one real transaction
    # --------------------------------------------------------

    data_file = (

        BASE_DIR
        / "data"
        / "processed"
        / "transactions_clean.csv"

    )


    df = pd.read_csv(
        data_file,
        nrows=1
    )


    transaction = (
        df.iloc[0]
        .to_dict()
    )


    # --------------------------------------------------------
    # Rename original columns
    # --------------------------------------------------------

    transaction[
        "transaction_time"
    ] = transaction.pop(
        "Time",
        0
    )


    transaction[
        "amount"
    ] = transaction.pop(
        "Amount",
        0
    )


    transaction[
        "is_fraud"
    ] = transaction.pop(
        "Class",
        0
    )


    for i in range(1, 29):

        old_name = f"V{i}"

        new_name = f"v{i}"

        if old_name in transaction:

            transaction[
                new_name
            ] = transaction.pop(
                old_name
            )


    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    engine = SentinelPayRiskEngine()


    result = engine.predict(
        transaction
    )


    print("\nTRANSACTION RISK RESULT")
    print("-" * 50)


    for key, value in result.items():

        print(
            f"{key}: {value}"
        )


    print("\n" + "=" * 70)
    print("Risk engine test completed.")
    print("=" * 70)