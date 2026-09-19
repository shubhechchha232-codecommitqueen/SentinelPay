from pathlib import Path
import os
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


# ============================================================
# SENTINELPAY - POSTGRESQL DATA LOADER
# ============================================================

print("=" * 70)
print("       SENTINELPAY - POSTGRESQL DATA LOADER")
print("=" * 70)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ENV_FILE = BASE_DIR / ".env"

CSV_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "transactions_clean.csv"
)


# ============================================================
# 2. LOAD .ENV
# ============================================================

print("\nLoading environment configuration...")

load_dotenv(
    dotenv_path=ENV_FILE,
    override=True
)


# ============================================================
# 3. READ DATABASE SETTINGS
# ============================================================

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


print("\nDatabase configuration:")
print("HOST     :", repr(DB_HOST))
print("PORT     :", repr(DB_PORT))
print("DATABASE :", repr(DB_NAME))
print("USER     :", repr(DB_USER))
print(
    "PASSWORD :",
    "LOADED" if DB_PASSWORD else "NOT FOUND"
)


# ============================================================
# 4. VALIDATE SETTINGS
# ============================================================

if not DB_NAME:
    raise ValueError("DB_NAME is missing from .env")

if not DB_USER:
    raise ValueError("DB_USER is missing from .env")

if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD is missing from .env")


# ============================================================
# 5. CHECK DATASET
# ============================================================

print("\nChecking processed dataset...")

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Processed dataset not found:\n{CSV_FILE}"
    )

print("Dataset found:")
print(CSV_FILE)


# ============================================================
# 6. CREATE POSTGRESQL CONNECTION
# ============================================================

print("\nConnecting to PostgreSQL...")


# IMPORTANT:
# Host is explicitly set to localhost.
# We are NOT constructing a connection string manually.

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
# 7. TEST CONNECTION
# ============================================================

try:

    with engine.connect() as connection:

        connection.execute(
            text("SELECT 1")
        )

    print("PostgreSQL connection successful.")

except Exception as error:

    print("\nDATABASE CONNECTION FAILED")
    print("-" * 50)
    print(str(error))
    print("-" * 50)

    raise


# ============================================================
# 8. LOAD CSV IN CHUNKS
# ============================================================

print("\nLoading processed dataset...")

CHUNK_SIZE = 5000

total_rows = 0
chunk_number = 0


# ============================================================
# 9. READ CSV
# ============================================================

for chunk in pd.read_csv(
    CSV_FILE,
    chunksize=CHUNK_SIZE
):

    chunk_number += 1


    # ========================================================
    # RENAME COLUMNS
    # ========================================================

    chunk = chunk.rename(
        columns={
            "Time": "transaction_time",
            "Amount": "amount",
            "Class": "is_fraud"
        }
    )


    # ========================================================
    # RENAME V1-V28
    # ========================================================

    rename_dictionary = {}

    for i in range(1, 29):

        rename_dictionary[
            f"V{i}"
        ] = f"v{i}"

    chunk = chunk.rename(
        columns=rename_dictionary
    )


    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = [

        "transaction_id",

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

        "pca_abs_mean",

        "is_fraud"
    ]


    # ========================================================
    # CHECK COLUMNS
    # ========================================================

    missing_columns = [
        column
        for column in required_columns
        if column not in chunk.columns
    ]

    if missing_columns:

        print("\nMissing columns:")
        print(missing_columns)

        raise ValueError(
            "Required columns are missing."
        )


    # ========================================================
    # SELECT REQUIRED COLUMNS
    # ========================================================

    chunk = chunk[
        required_columns
    ]


    # ========================================================
    # INSERT INTO POSTGRESQL
    # ========================================================

    chunk.to_sql(
        name="transactions",
        con=engine,
        if_exists="append",
        index=False,
        method="multi"
    )


    # ========================================================
    # UPDATE COUNTER
    # ========================================================

    total_rows += len(chunk)


    # ========================================================
    # PROGRESS
    # ========================================================

    print(
        f"Chunk {chunk_number:03d} loaded | "
        f"Rows inserted: {total_rows:,}"
    )


# ============================================================
# 10. VERIFY DATABASE
# ============================================================

print("\nVerifying PostgreSQL data...")

with engine.connect() as connection:

    result = connection.execute(
        text(
            "SELECT COUNT(*) FROM transactions"
        )
    )

    database_count = result.scalar()


# ============================================================
# 11. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("              DATA LOAD COMPLETE")
print("=" * 70)

print(
    f"\nRows processed by Python : {total_rows:,}"
)

print(
    f"Rows present in database : {database_count:,}"
)


if database_count == total_rows:

    print(
        "\nDATABASE VERIFICATION: SUCCESS"
    )

else:

    print(
        "\nWARNING: ROW COUNT MISMATCH"
    )


print(
    "\nPostgreSQL transactions table is ready."
)

print("=" * 70)