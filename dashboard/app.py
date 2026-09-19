import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import json
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# SENTINELPAY - AI FRAUD & ANOMALY DETECTION DASHBOARD
# ============================================================

st.set_page_config(
    page_title="SentinelPay Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"

DATA_PATHS = [
    "data/processed/transactions_clean.csv",
    "./data/processed/transactions_clean.csv",
    "../data/processed/transactions_clean.csv"
]

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .metric-card {
        background: #171a21;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30343b;
        text-align: center;
    }

    .metric-title {
        color: #9ca3af;
        font-size: 14px;
    }

    .metric-value {
        color: white;
        font-size: 30px;
        font-weight: 700;
    }

    .online {
        background-color: #123b27;
        color: #3cff83;
        padding: 14px 20px;
        border-radius: 10px;
        font-size: 16px;
        margin-bottom: 20px;
    }

    .offline {
        background-color: #442126;
        color: #ff5f5f;
        padding: 14px 20px;
        border-radius: 10px;
        font-size: 16px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_dataset():

    for path in DATA_PATHS:

        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df
            except Exception:
                pass

    return None


df = load_dataset()

# ============================================================
# API FUNCTIONS
# ============================================================

def check_api():

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )

        if response.status_code == 200:
            return True, response.json()

    except Exception:
        pass

    return False, None


def get_model_info():

    try:
        response = requests.get(
            f"{API_URL}/model-info",
            timeout=5
        )

        if response.status_code == 200:
            return response.json()

    except Exception:
        pass

    return None


def predict_transaction(payload):

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=20
        )

        if response.status_code == 200:
            return response.json()

        return {
            "error": response.text
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# ============================================================
# API STATUS
# ============================================================

api_online, health_data = check_api()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h1 style="margin-bottom:5px;">🛡️ SentinelPay</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Navigation")

    page = st.radio(
        "",
        [
            "🏠 Dashboard",
            "🔍 Transaction Analysis",
            "📊 Risk Analysis",
            "🤖 Model Information"
        ]
    )

    st.markdown("---")

    st.markdown("### System")

    st.markdown(
        """
        <div style="
            background:#29486b;
            padding:15px;
            border-radius:10px;
            line-height:2;
        ">
        <b>Fraud Detection</b><br>
        Anomaly Detection<br>
        Machine Learning<br>
        Autoencoder<br>
        Risk Engine<br>
        FastAPI
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <h1>🛡️ SentinelPay</h1>
    <p style="font-size:18px;color:#7890ad;">
    AI-Powered Real-Time Fraud & Anomaly Detection Platform
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ============================================================
# API STATUS
# ============================================================

if api_online:

    st.markdown(
        """
        <div class="online">
        🟢 SentinelPay API is <b>ONLINE</b>
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class="offline">
        🔴 SentinelPay API is OFFLINE.
        Start it using:
        <b>python src/api.py</b>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# CALCULATE DATASET STATISTICS
# ============================================================

if df is not None:

    total_transactions = len(df)

    fraud_column = None

    possible_fraud_columns = [
        "Class",
        "class",
        "Fraud",
        "fraud",
        "is_fraud",
        "fraud_label"
    ]

    for column in possible_fraud_columns:

        if column in df.columns:
            fraud_column = column
            break

    if fraud_column:

        fraud_cases = int(
            pd.to_numeric(
                df[fraud_column],
                errors="coerce"
            ).fillna(0).sum()
        )

    else:

        fraud_cases = 0

    fraud_rate = (
        fraud_cases / total_transactions * 100
        if total_transactions > 0
        else 0
    )

else:

    total_transactions = 0
    fraud_cases = 0
    fraud_rate = 0


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Transactions",
            f"{total_transactions:,}"
        )

    with col2:

        st.metric(
            "Fraud Cases",
            f"{fraud_cases:,}"
        )

    with col3:

        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.4f}%"
        )

    with col4:

        st.metric(
            "Models",
            "3"
        )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🔍 Real-Time Transaction Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter transaction features below and SentinelPay will calculate "
        "the fraud and anomaly risk."
    )

    col1, col2 = st.columns(2)

    with col1:

        transaction_time = st.number_input(
            "Transaction Time",
            value=0.06,
            format="%.6f"
        )

    with col2:

        transaction_id = st.text_input(
            "Transaction ID",
            value="TXN-DEMO-001"
        )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    with st.expander("Advanced Transaction Features"):

        st.write(
            "Enter the PCA transaction features used by the fraud models."
        )

        feature_values = {}

        columns = st.columns(4)

        for i in range(1, 29):

            with columns[(i - 1) % 4]:

                feature_values[f"V{i}"] = st.number_input(
                    f"V{i}",
                    value=0.0,
                    format="%.6f",
                    key=f"dashboard_v{i}"
                )

    if st.button(
        "🚨 ANALYZE TRANSACTION",
        use_container_width=True
    ):

        payload = {
            "Time": transaction_time,
            "Amount": amount
        }

        payload.update(feature_values)

        result = predict_transaction(payload)

        if "error" in result:

            st.error(
                "Could not connect to SentinelPay API."
            )

            st.code(result["error"])

        else:

            st.success(
                "Transaction analyzed successfully."
            )

            risk = result.get(
                "risk_analysis",
                {}
            )

            transaction = result.get(
                "transaction",
                {}
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Risk Score",
                    f"{risk.get('risk_score', 0):.2f}"
                )

            with col2:

                st.metric(
                    "Risk Level",
                    risk.get(
                        "risk_level",
                        "UNKNOWN"
                    )
                )

            with col3:

                fraud_probability = risk.get(
                    "fraud_probability",
                    0
                )

                st.metric(
                    "Fraud Probability",
                    f"{fraud_probability * 100:.2f}%"
                )

            with col4:

                st.metric(
                    "Anomaly Probability",
                    f"{risk.get('anomaly_probability', 0):.2f}"
                )

            st.markdown("---")

            st.subheader("Risk Assessment")

            score = float(
                risk.get(
                    "risk_score",
                    0
                )
            )

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={
                        "text": "Risk Score"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 1]
                        }
                    }
                )
            )

            fig.update_layout(
                height=350
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            reasons = risk.get(
                "reasons",
                []
            )

            if reasons:

                st.subheader("Detection Reasons")

                for reason in reasons:

                    st.write(
                        f"• {reason}"
                    )

            st.subheader("Model Analysis")

            model_data = {

                "Metric": [
                    "Fraud Probability",
                    "Anomaly Probability",
                    "Reconstruction Error"
                ],

                "Value": [

                    risk.get(
                        "fraud_probability",
                        0
                    ),

                    risk.get(
                        "anomaly_probability",
                        0
                    ),

                    risk.get(
                        "reconstruction_error",
                        0
                    )
                ]
            }

            st.dataframe(
                pd.DataFrame(model_data),
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# TRANSACTION ANALYSIS
# ============================================================

elif page == "🔍 Transaction Analysis":

    st.markdown(
        '<div class="section-title">🔍 Transaction Analysis</div>',
        unsafe_allow_html=True
    )

    if df is None:

        st.error(
            "Processed transaction dataset could not be found."
        )

    else:

        st.write(
            "Explore the transaction dataset used by SentinelPay."
        )

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("Transaction Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Transactions",
                f"{len(df):,}"
            )

        with col2:

            if "Amount" in df.columns:

                st.metric(
                    "Average Amount",
                    f"${df['Amount'].mean():,.2f}"
                )

        with col3:

            if "Amount" in df.columns:

                st.metric(
                    "Maximum Amount",
                    f"${df['Amount'].max():,.2f}"
                )

        with col4:

            if "Amount" in df.columns:

                st.metric(
                    "Minimum Amount",
                    f"${df['Amount'].min():,.2f}"
                )

        if "Amount" in df.columns:

            st.markdown("---")

            st.subheader(
                "Transaction Amount Distribution"
            )

            fig = px.histogram(
                df,
                x="Amount",
                nbins=50,
                title="Transaction Amount Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# RISK ANALYSIS
# ============================================================

elif page == "📊 Risk Analysis":

    st.markdown(
        '<div class="section-title">📊 Risk Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Dataset-level fraud and transaction risk analytics."
    )

    if df is None:

        st.error(
            "Dataset not found."
        )

    else:

        # ----------------------------------------------------
        # FRAUD / LEGITIMATE COUNTS
        # ----------------------------------------------------

        if fraud_column:

            fraud_values = (
                df[fraud_column]
                .value_counts()
                .reset_index()
            )

            fraud_values.columns = [
                "Class",
                "Count"
            ]

            st.subheader(
                "Fraud vs Legitimate Transactions"
            )

            col1, col2 = st.columns(2)

            with col1:

                fig = px.pie(
                    fraud_values,
                    names="Class",
                    values="Count",
                    title="Transaction Class Distribution",
                    hole=0.45
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with col2:

                fig = px.bar(
                    fraud_values,
                    x="Class",
                    y="Count",
                    title="Fraud / Legitimate Transaction Count"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        else:

            st.info(
                "Fraud label column was not detected in the dataset."
            )

        # ----------------------------------------------------
        # AMOUNT ANALYSIS
        # ----------------------------------------------------

        if "Amount" in df.columns:

            st.markdown("---")

            st.subheader(
                "Transaction Amount Analysis"
            )

            col1, col2 = st.columns(2)

            with col1:

                fig = px.histogram(
                    df,
                    x="Amount",
                    nbins=60,
                    title="Transaction Amount Distribution"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with col2:

                fig = px.box(
                    df,
                    y="Amount",
                    title="Transaction Amount Box Plot"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # ----------------------------------------------------
        # TIME ANALYSIS
        # ----------------------------------------------------

        if "Time" in df.columns:

            st.markdown("---")

            st.subheader(
                "Transaction Time Analysis"
            )

            time_data = df.copy()

            time_data["Time_Bin"] = (
                time_data["Time"] // 1000
            ) * 1000

            time_counts = (
                time_data
                .groupby("Time_Bin")
                .size()
                .reset_index(name="Transactions")
            )

            fig = px.line(
                time_counts,
                x="Time_Bin",
                y="Transactions",
                title="Transaction Activity Over Time"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # FRAUD AMOUNT ANALYSIS
        # ----------------------------------------------------

        if fraud_column and "Amount" in df.columns:

            st.markdown("---")

            st.subheader(
                "Fraud Transaction Amount Analysis"
            )

            fraud_df = df[
                pd.to_numeric(
                    df[fraud_column],
                    errors="coerce"
                ) == 1
            ]

            if len(fraud_df) > 0:

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Fraud Transactions",
                        f"{len(fraud_df):,}"
                    )

                with col2:

                    st.metric(
                        "Average Fraud Amount",
                        f"${fraud_df['Amount'].mean():,.2f}"
                    )

                with col3:

                    st.metric(
                        "Maximum Fraud Amount",
                        f"${fraud_df['Amount'].max():,.2f}"
                    )

                fig = px.histogram(
                    fraud_df,
                    x="Amount",
                    nbins=40,
                    title="Fraud Transaction Amount Distribution"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # ----------------------------------------------------
        # DATASET SUMMARY
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "Risk Dataset Summary"
        )

        summary_data = {

            "Metric": [
                "Total Transactions",
                "Fraud Cases",
                "Fraud Rate",
                "Number of Features"
            ],

            "Value": [
                f"{len(df):,}",
                f"{fraud_cases:,}",
                f"{fraud_rate:.4f}%",
                f"{len(df.columns):,}"
            ]
        }

        st.dataframe(
            pd.DataFrame(summary_data),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "🤖 Model Information":

    st.markdown(
        '<div class="section-title">🤖 Model Information</div>',
        unsafe_allow_html=True
    )

    model_info = get_model_info()

    if model_info:

        st.success(
            "Model information loaded successfully."
        )

        st.json(model_info)

    else:

        st.warning(
            "Could not retrieve model information from API."
        )

    st.markdown("---")

    st.subheader(
        "SentinelPay AI Architecture"
    )

    architecture = pd.DataFrame({

        "Component": [

            "Random Forest",
            "XGBoost",
            "Autoencoder",
            "Risk Engine",
            "FastAPI",
            "Streamlit"
        ],

        "Purpose": [

            "Supervised fraud classification",
            "Gradient boosting fraud detection",
            "Unsupervised anomaly detection",
            "Risk score aggregation",
            "Real-time model serving",
            "Interactive analytics dashboard"
        ]
    })

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader(
        "Project Pipeline"
    )

    st.code(
        """
Transaction
     │
     ▼
Feature Processing
     │
     ├───────────────┐
     ▼               ▼
Random Forest     XGBoost
     │               │
     └───────┬───────┘
             │
             ▼
       Fraud Probability
             │
             ▼
       Autoencoder
             │
             ▼
       Anomaly Score
             │
             ▼
        Risk Engine
             │
             ▼
      Final Risk Score
             │
             ▼
          FastAPI
             │
             ▼
        Streamlit UI
        """,
        language="text"
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "SentinelPay — AI-powered real-time fraud and anomaly detection system"
)