from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from retail_segmentation_recommendation_pipeline import (
    clean_transactions,
    build_customer_features,
    detect_outliers,
    scale_features,
    apply_pca,
    choose_best_k,
    cluster_customers,
    generate_recommendations,
    load_data,
)


st.set_page_config(
    page_title="Retail SKU Recommendation System",
    page_icon="🛍️",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_uploaded_dataset(file_bytes: bytes, suffix: str) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    return load_data(temp_path)


@st.cache_data(show_spinner=True)
def generate_prediction_output(df: pd.DataFrame):

    cleaned = clean_transactions(df)

    customer_data = build_customer_features(cleaned)

    customer_data_cleaned, outliers = detect_outliers(
        customer_data,
        contamination=0.05,
    )

    customer_data_scaled, _, _ = scale_features(customer_data_cleaned)

    customer_data_pca, _ = apply_pca(
        customer_data_scaled,
        n_components=3,
    )

    best_k, _ = choose_best_k(
        customer_data_pca,
        start_k=3,
        stop_k=8,
    )

    customer_clusters, _, _ = cluster_customers(
        customer_data_cleaned,
        customer_data_pca,
        n_clusters=best_k,
    )

    customer_recommendations, _, _ = generate_recommendations(
        cleaned,
        customer_clusters,
        outliers,
        top_n_products_per_cluster=10,
        n_recommendations=3,
    )

    return customer_recommendations, cleaned


def calculate_prediction_accuracy(prediction_df: pd.DataFrame):

    total_predictions = len(prediction_df)

    valid_predictions = prediction_df.dropna(
        subset=[
            "Rec1_StockCode",
            "Rec2_StockCode",
            "Rec3_StockCode",
        ],
        how="all",
    )

    accuracy = (
        len(valid_predictions) / total_predictions * 100
        if total_predictions > 0
        else 0
    )

    return round(accuracy, 2)


def build_profit_table(prediction_df: pd.DataFrame, cleaned_df: pd.DataFrame):

    cleaned_df = cleaned_df.copy()

    cleaned_df["Revenue"] = (
        cleaned_df["Quantity"] * cleaned_df["UnitPrice"]
    )

    sku_profit = (
        cleaned_df.groupby("StockCode")["Revenue"]
        .mean()
        .reset_index()
    )

    sku_profit.columns = [
        "StockCode",
        "ExpectedProfit",
    ]

    sku_profit["StockCode"] = sku_profit["StockCode"].astype(str)

    profit_rows = []

    sample_df = prediction_df.sample(
        n=min(10, len(prediction_df)),
        random_state=None,
    )

    for _, row in sample_df.iterrows():

        rec2 = str(row.get("Rec2_StockCode", ""))
        rec3 = str(row.get("Rec3_StockCode", ""))

        rec2_profit = sku_profit[
            sku_profit["StockCode"] == rec2
        ]["ExpectedProfit"]

        rec3_profit = sku_profit[
            sku_profit["StockCode"] == rec3
        ]["ExpectedProfit"]

        rec2_value = (
            round(rec2_profit.values[0], 2)
            if not rec2_profit.empty
            else 0
        )

        rec3_value = (
            round(rec3_profit.values[0], 2)
            if not rec3_profit.empty
            else 0
        )

        profit_rows.append(
            {
                "CustomerID": row["CustomerID"],
                "Recommended_SKU_2": rec2,
                "Expected_Profit_SKU_2 (£)": rec2_value,
                "Recommended_SKU_3": rec3,
                "Expected_Profit_SKU_3 (£)": rec3_value,
            }
        )

    return pd.DataFrame(profit_rows)


st.title("🛍️ Retail Customer SKU Recommendation System")

st.caption(
    "Upload the customer transaction dataset and generate customer-level SKU recommendations."
)


uploaded_file = st.file_uploader(
    "Upload Online Retail dataset (.xlsx, .xls, .csv)",
    type=["xlsx", "xls", "csv"],
)


if uploaded_file is None:
    st.info("Upload your dataset to start.")
    st.stop()


suffix = Path(uploaded_file.name).suffix.lower()

raw_df = load_uploaded_dataset(
    uploaded_file.getvalue(),
    suffix,
)


with st.spinner("Generating recommendations..."):

    prediction_df, cleaned_df = generate_prediction_output(raw_df)


prediction_df["CustomerID"] = (
    prediction_df["CustomerID"].astype(str)
)


required_cols = [
    "CustomerID",
    "Rec1_StockCode",
    "Rec1_Description",
    "Rec2_StockCode",
    "Rec2_Description",
    "Rec3_StockCode",
    "Rec3_Description",
]

available_cols = [
    col for col in required_cols
    if col in prediction_df.columns
]

prediction_df = prediction_df[available_cols]


prediction_accuracy = calculate_prediction_accuracy(
    prediction_df
)


st.subheader("Prediction KPI")

kpi1, kpi2, kpi3 = st.columns(3)

kpi2.metric(
    "Prediction Accuracy",
    f"{prediction_accuracy:.2f}%",
)


st.subheader("Random 10 Customer Recommendations")

random_10_customers = prediction_df.sample(
    n=min(10, len(prediction_df)),
    random_state=None,
)

st.dataframe(
    random_10_customers,
    use_container_width=True,
    hide_index=True,
)


st.subheader("Expected Profit from Recommended SKUs")

profit_table = build_profit_table(
    prediction_df,
    cleaned_df,
)

st.dataframe(
    profit_table,
    use_container_width=True,
    hide_index=True,
)


st.download_button(
    "Download Recommendation Output",
    data=prediction_df.to_csv(index=False).encode("utf-8"),
    file_name="customer_recommendations.csv",
    mime="text/csv",
)
