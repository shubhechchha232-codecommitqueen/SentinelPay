import pandas as pd
from pathlib import Path

# SentinelPay dataset location
DATA_PATH = Path("data/raw/creditcard.csv")

print("=" * 70)
print("        SENTINELPAY - DATASET INSPECTION")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")

print("\n1. DATASET SHAPE")
print("-" * 40)
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

print("\n2. COLUMN NAMES")
print("-" * 40)
print(df.columns.tolist())

print("\n3. FIRST 5 TRANSACTIONS")
print("-" * 40)
print(df.head())

print("\n4. MISSING VALUES")
print("-" * 40)
print(df.isnull().sum().sum())

print("\n5. FRAUD DISTRIBUTION")
print("-" * 40)
print(df["Class"].value_counts())

print("\n6. FRAUD PERCENTAGE")
print("-" * 40)
fraud_percentage = df["Class"].mean() * 100
print(f"{fraud_percentage:.4f}%")

print("\n7. DUPLICATE ROWS")
print("-" * 40)
print(df.duplicated().sum())

print("\n8. DATA TYPES")
print("-" * 40)
print(df.dtypes)

print("\n" + "=" * 70)
print("              INSPECTION COMPLETE")
print("=" * 70)