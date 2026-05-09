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
    page_title="Retail SKU Recommendation Output",
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


def calculate_business_kpis(prediction_df: pd.DataFrame, cleaned_df: pd.DataFrame):
    cleaned_df = cleaned_df.copy()
    cleaned_df["StockCode"] = cleaned_df["StockCode"].astype(str)
    cleaned_df["Revenue"] = cleaned_df["Quantity"] * cleaned_df["UnitPrice"]

    avg_sku_price = (
        cleaned_df.groupby("StockCode")["UnitPrice"]
        .mean()
        .to_dict()
    )

    rec_cols = [
        "Rec1_StockCode",
        "Rec2_StockCode",
        "Rec3_StockCode",
    ]

    value_df = prediction_df.copy()

    total_opportunity = 0

    for col in rec_cols:
        if col in value_df.columns:
            value_df[col] = value_df[col].astype(str)
            total_opportunity += value_df[col].map(avg_sku_price).fillna(0).sum()

    total_customers = prediction_df["CustomerID"].nunique()

    avg_recommendation_value = (
        total_opportunity / total_customers
        if total_customers > 0
        else 0
    )

    historical_revenue = cleaned_df["Revenue"].sum()

    return historical_revenue, total_opportunity, avg_recommendation_value


st.title("🛍️ Retail Customer SKU Recommendation System")

st.caption(
    "Upload the customer transaction dataset and view a random sample of customer-level SKU recommendations."
)


uploaded_file = st.file_uploader(
    "Upload Online Retail dataset (.xlsx, .xls, .csv)",
    type=["xlsx", "xls", "csv"],
)


if uploaded_file is None:
    st.info("Upload your dataset to generate customer SKU recommendations.")
    st.stop()


suffix = Path(uploaded_file.name).suffix.lower()
raw_df = load_uploaded_dataset(uploaded_file.getvalue(), suffix)


with st.spinner("Generating customer SKU recommendations..."):
    prediction_df, cleaned_df = generate_prediction_output(raw_df)


prediction_df["CustomerID"] = prediction_df["CustomerID"].astype(str)

required_cols = [
    "CustomerID",
    "Rec1_StockCode",
    "Rec1_Description",
    "Rec2_StockCode",
    "Rec2_Description",
    "Rec3_StockCode",
    "Rec3_Description",
]

available_cols = [col for col in required_cols if col in prediction_df.columns]
prediction_df = prediction_df[available_cols]


historical_revenue, total_opportunity, avg_recommendation_value = calculate_business_kpis(
    prediction_df,
    cleaned_df,
)


st.subheader("Business Recommendation KPIs")

kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric(
    "Historical Sales Revenue",
    f"£{historical_revenue:,.0f}",
)

kpi2.metric(
    "Estimated Revenue Opportunity",
    f"£{total_opportunity:,.0f}",
)

kpi3.metric(
    "Avg Recommendation Value / Customer",
    f"£{avg_recommendation_value:,.2f}",
)


st.subheader("Random 10 Customer SKU Recommendations")

sample_size = min(10, len(prediction_df))

random_10_customers = prediction_df.sample(
    n=sample_size,
    random_state=None,
)

st.dataframe(
    random_10_customers,
    use_container_width=True,
    hide_index=True,
)


st.download_button(
    "Download Full Recommendation Output",
    data=prediction_df.to_csv(index=False).encode("utf-8"),
    file_name="customer_sku_recommendations.csv",
    mime="text/csv",
)
