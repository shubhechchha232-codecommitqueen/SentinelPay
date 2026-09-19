from pathlib import Path
import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# SENTINELPAY - PYTORCH AUTOENCODER
# UNSUPERVISED ANOMALY DETECTION
# ============================================================

print("=" * 70)
print("       SENTINELPAY - PYTORCH ANOMALY DETECTION")
print("=" * 70)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

ENV_FILE = BASE_DIR / ".env"

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD ENVIRONMENT
# ============================================================

print("\n[1/8] Loading environment...")

load_dotenv(
    dotenv_path=ENV_FILE,
    override=True
)

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ============================================================
# 3. POSTGRESQL CONNECTION
# ============================================================

print("\n[2/8] Connecting to PostgreSQL...")


DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host="localhost",
    port=5432,
    database=DB_NAME
)


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n[3/8] Loading transactions...")


df = pd.read_sql(
    "SELECT * FROM transactions",
    engine
)


print(
    f"Transactions loaded: {len(df):,}"
)


# ============================================================
# 5. FEATURES
# ============================================================

print("\n[4/8] Preparing anomaly-detection features...")


FEATURE_COLUMNS = [

    "transaction_time",

    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v7",
    "v8",
    "v9",
    "v10",
    "v11",
    "v12",
    "v13",
    "v14",
    "v15",
    "v16",
    "v17",
    "v18",
    "v19",
    "v20",
    "v21",
    "v22",
    "v23",
    "v24",
    "v25",
    "v26",
    "v27",
    "v28",

    "amount",

    "amount_log",

    "amount_zscore",

    "transaction_frequency_24h",

    "avg_transaction_amount",

    "amount_deviation",

    "pca_mean",

    "pca_std",

    "pca_abs_mean"
]


X = df[
    FEATURE_COLUMNS
].copy()


y = df[
    "is_fraud"
].astype(int)


# Replace problematic values

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median()
)


# ============================================================
# 6. TRAIN ONLY ON NORMAL TRANSACTIONS
# ============================================================

print("\n[5/8] Selecting normal transactions...")


normal_data = X[
    y == 0
].copy()


print(
    f"Normal transactions available: "
    f"{len(normal_data):,}"
)


# Keep a manageable training sample
# This makes CPU training faster on a laptop.

if len(normal_data) > 150000:

    normal_data = normal_data.sample(
        n=150000,
        random_state=42
    )


print(
    f"Autoencoder training samples: "
    f"{len(normal_data):,}"
)


# ============================================================
# 7. SCALE DATA
# ============================================================

scaler = StandardScaler()


X_normal_scaled = scaler.fit_transform(
    normal_data
)


X_normal_scaled = X_normal_scaled.astype(
    np.float32
)


# ============================================================
# 8. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_validation = train_test_split(

    X_normal_scaled,

    test_size=0.15,

    random_state=42
)


# Convert to PyTorch tensors

train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

validation_tensor = torch.tensor(
    X_validation,
    dtype=torch.float32
)


train_dataset = TensorDataset(
    train_tensor,
    train_tensor
)


validation_dataset = TensorDataset(
    validation_tensor,
    validation_tensor
)


train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=512,
    shuffle=False
)


# ============================================================
# 9. AUTOENCODER ARCHITECTURE
# ============================================================

print("\n[6/8] Building PyTorch Autoencoder...")


input_dimension = len(
    FEATURE_COLUMNS
)


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
# 10. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    f"Training device: {device}"
)


model = FraudAutoencoder(
    input_dimension
).to(device)


# ============================================================
# 11. OPTIMIZER / LOSS
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=0.001,

    weight_decay=1e-5
)


# ============================================================
# 12. TRAINING
# ============================================================

print("\nTraining Autoencoder...")


EPOCHS = 20


for epoch in range(
    EPOCHS
):

    model.train()

    training_loss = 0.0


    for batch_x, _ in train_loader:

        batch_x = batch_x.to(
            device
        )


        optimizer.zero_grad()


        reconstructed = model(
            batch_x
        )


        loss = criterion(
            reconstructed,
            batch_x
        )


        loss.backward()


        optimizer.step()


        training_loss += (
            loss.item()
            * batch_x.size(0)
        )


    training_loss /= len(
        train_loader.dataset
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    validation_loss = 0.0


    with torch.no_grad():

        for batch_x, _ in validation_loader:

            batch_x = batch_x.to(
                device
            )


            reconstructed = model(
                batch_x
            )


            loss = criterion(
                reconstructed,
                batch_x
            )


            validation_loss += (
                loss.item()
                * batch_x.size(0)
            )


    validation_loss /= len(
        validation_loader.dataset
    )


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {training_loss:.6f} | "
        f"Validation Loss: {validation_loss:.6f}"
    )


# ============================================================
# 13. CALCULATE ANOMALY THRESHOLD
# ============================================================

print("\n[7/8] Calculating anomaly threshold...")


model.eval()


validation_errors = []


with torch.no_grad():

    for batch_x, _ in validation_loader:

        batch_x = batch_x.to(
            device
        )


        reconstructed = model(
            batch_x
        )


        errors = torch.mean(
            (reconstructed - batch_x) ** 2,
            dim=1
        )


        validation_errors.extend(
            errors.cpu().numpy()
        )


validation_errors = np.array(
    validation_errors
)


# High reconstruction error = unusual transaction

threshold = np.percentile(
    validation_errors,
    99.5
)


print(
    f"Anomaly threshold: {threshold:.6f}"
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

print("\n[8/8] Saving Autoencoder...")


torch.save(

    {
        "model_state_dict":
            model.state_dict(),

        "input_dimension":
            input_dimension,

        "feature_columns":
            FEATURE_COLUMNS,

        "anomaly_threshold":
            float(threshold)

    },

    MODEL_DIR /
    "fraud_autoencoder.pth"
)


# Save scaler

import joblib

joblib.dump(

    scaler,

    MODEL_DIR /
    "autoencoder_scaler.pkl"
)


# Save configuration

autoencoder_config = {

    "input_dimension":
        input_dimension,

    "threshold":
        float(threshold),

    "epochs":
        EPOCHS,

    "batch_size":
        512,

    "latent_dimension":
        16,

    "training_samples":
        len(normal_data)

}


with open(

    MODEL_DIR /
    "autoencoder_config.json",

    "w"

) as file:

    json.dump(
        autoencoder_config,
        file,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("        AUTOENCODER TRAINING COMPLETE")
print("=" * 70)

print(
    "\nSaved:"
)

print(
    MODEL_DIR /
    "fraud_autoencoder.pth"
)

print(
    MODEL_DIR /
    "autoencoder_scaler.pkl"
)

print(
    MODEL_DIR /
    "autoencoder_config.json"
)

print(
    "\nAnomaly detection layer is ready."
)

print("=" * 70)