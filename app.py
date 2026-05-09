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


def build_revenue_opportunity_table(
    random_customers_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
) -> pd.DataFrame:
    cleaned_df = cleaned_df.copy()

    cleaned_df["StockCode"] = cleaned_df["StockCode"].astype(str)

    cleaned_df = cleaned_df[
        cleaned_df["Quantity"] > 0
    ]

    sku_stats = (
        cleaned_df.groupby("StockCode")
        .agg(
            Expected_Qty=("Quantity", "mean"),
            Avg_UnitPrice=("UnitPrice", "mean"),
        )
        .reset_index()
    )

    sku_lookup = sku_stats.set_index("StockCode").to_dict("index")

    rows = []

    for _, row in random_customers_df.iterrows():
        rec1_sku = str(row.get("Rec1_StockCode", ""))
        rec2_sku = str(row.get("Rec2_StockCode", ""))
        rec3_sku = str(row.get("Rec3_StockCode", ""))

        rec1_stats = sku_lookup.get(rec1_sku, {})
        rec2_stats = sku_lookup.get(rec2_sku, {})
        rec3_stats = sku_lookup.get(rec3_sku, {})

        rec1_qty = round(rec1_stats.get("Expected_Qty", 0), 0)
        rec2_qty = round(rec2_stats.get("Expected_Qty", 0), 0)
        rec3_qty = round(rec3_stats.get("Expected_Qty", 0), 0)

        rec1_price = round(rec1_stats.get("Avg_UnitPrice", 0), 2)
        rec2_price = round(rec2_stats.get("Avg_UnitPrice", 0), 2)
        rec3_price = round(rec3_stats.get("Avg_UnitPrice", 0), 2)

        rec1_revenue = rec1_qty * rec1_price
        rec2_revenue = rec2_qty * rec2_price
        rec3_revenue = rec3_qty * rec3_price

        total_revenue = (
            rec1_revenue
            + rec2_revenue
            + rec3_revenue
        )

        rows.append(
            {
                "CustomerID": row["CustomerID"],

                "Rec1_SKU": rec1_sku,
                "Rec1_Description": row.get("Rec1_Description", ""),
                "Rec1_Expected_Qty": rec1_qty,
                "Rec1_UnitPrice": rec1_price,
                "Rec1_Revenue": round(rec1_revenue, 2),

                "Rec2_SKU": rec2_sku,
                "Rec2_Description": row.get("Rec2_Description", ""),
                "Rec2_Expected_Qty": rec2_qty,
                "Rec2_UnitPrice": rec2_price,
                "Rec2_Revenue": round(rec2_revenue, 2),

                "Rec3_SKU": rec3_sku,
                "Rec3_Description": row.get("Rec3_Description", ""),
                "Rec3_Expected_Qty": rec3_qty,
                "Rec3_UnitPrice": rec3_price,
                "Rec3_Revenue": round(rec3_revenue, 2),

                "Total_Expected_Revenue": round(total_revenue, 2),
            }
        )

    revenue_df = pd.DataFrame(rows)

    revenue_df = revenue_df.sort_values(
        by="Total_Expected_Revenue",
        ascending=False,
    )

    return revenue_df


st.title("🛍️ Retail Customer SKU Recommendation System")

st.caption(
    "Upload the customer transaction dataset and generate customer-level SKU recommendations with expected revenue opportunity."
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

available_cols = [
    col for col in required_cols
    if col in prediction_df.columns
]

prediction_df = prediction_df[available_cols]


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


st.subheader("Expected Revenue Opportunity")

st.caption(
    "Expected revenue is calculated using the historical average quantity and average unit price for each recommended SKU from the uploaded dataset."
)

revenue_table = build_revenue_opportunity_table(
    random_10_customers,
    cleaned_df,
)

st.dataframe(
    revenue_table,
    use_container_width=True,
    hide_index=True,
)


total_revenue_opportunity = revenue_table["Total_Expected_Revenue"].sum()

st.info(
    f"Total expected revenue from the displayed 10 customers: "
    f"£{total_revenue_opportunity:,.2f}"
)


st.download_button(
    "Download Full Recommendation Output",
    data=prediction_df.to_csv(index=False).encode("utf-8"),
    file_name="customer_recommendations.csv",
    mime="text/csv",
)


st.download_button(
    "Download Revenue Opportunity Output",
    data=revenue_table.to_csv(index=False).encode("utf-8"),
    file_name="revenue_opportunity.csv",
    mime="text/csv",
)
