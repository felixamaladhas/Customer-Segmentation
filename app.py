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
    page_title="Customer SKU Recommendations",
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
def generate_prediction_output(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = clean_transactions(df)

    customer_data = build_customer_features(cleaned)
    customer_data_cleaned, outliers = detect_outliers(
        customer_data,
        contamination=0.05,
    )

    customer_data_scaled, _, _ = scale_features(customer_data_cleaned)
    customer_data_pca, _ = apply_pca(customer_data_scaled, n_components=3)

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

    return customer_recommendations


st.title("🛍️ Retail Customer SKU Recommendation System")
st.caption("Upload the customer transaction dataset and view recommended SKUs by CustomerID.")


uploaded_file = st.file_uploader(
    "Upload Online Retail dataset (.xlsx, .xls, .csv)",
    type=["xlsx", "xls", "csv"],
)


if uploaded_file is None:
    st.info("Upload your dataset to start.")
    st.stop()


suffix = Path(uploaded_file.name).suffix.lower()
raw_df = load_uploaded_dataset(uploaded_file.getvalue(), suffix)


with st.spinner("Generating customer SKU recommendations..."):
    prediction_df = generate_prediction_output(raw_df)


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


with st.sidebar:
    st.header("Customer Selection")

    customer_list = sorted(prediction_df["CustomerID"].dropna().unique())

    selected_customer = st.selectbox(
        "Select Customer Number",
        options=customer_list,
    )


st.subheader("Recommended SKUs")

customer_output = prediction_df[
    prediction_df["CustomerID"] == selected_customer
]


if customer_output.empty:
    st.warning("No SKU recommendations found for this customer.")
else:
    st.success(f"Showing recommended SKUs for CustomerID: {selected_customer}")

    st.dataframe(
        customer_output,
        use_container_width=True,
        hide_index=True,
    )


st.download_button(
    "Download Selected Customer Recommendations",
    data=customer_output.to_csv(index=False).encode("utf-8"),
    file_name=f"customer_{selected_customer}_recommendations.csv",
    mime="text/csv",
)
