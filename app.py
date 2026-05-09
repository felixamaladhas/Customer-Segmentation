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
def generate_prediction_output(
    df: pd.DataFrame,
    number_of_skus: int,
):
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
        top_n_products_per_cluster=max(10, number_of_skus),
        n_recommendations=number_of_skus,
    )

    return customer_recommendations, cleaned


def get_recommendation_columns(number_of_skus: int):
    cols = ["CustomerID"]

    for i in range(1, number_of_skus + 1):
        cols.extend(
            [
                f"Rec{i}_StockCode",
                f"Rec{i}_Description",
            ]
        )

    return cols


def build_revenue_opportunity_table(
    random_customers_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    number_of_skus: int,
) -> pd.DataFrame:
    cleaned_df = cleaned_df.copy()
    cleaned_df["StockCode"] = cleaned_df["StockCode"].astype(str)
    cleaned_df = cleaned_df[cleaned_df["Quantity"] > 0]

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
        output_row = {
            "CustomerID": row["CustomerID"],
        }

        total_revenue = 0

        for i in range(1, number_of_skus + 1):
            sku_col = f"Rec{i}_StockCode"
            desc_col = f"Rec{i}_Description"

            sku = str(row.get(sku_col, ""))
            desc = row.get(desc_col, "")

            sku_data = sku_lookup.get(sku, {})

            expected_qty = round(sku_data.get("Expected_Qty", 0), 0)
            avg_unit_price = round(sku_data.get("Avg_UnitPrice", 0), 2)
            revenue = round(expected_qty * avg_unit_price, 2)

            total_revenue += revenue

            output_row[f"Rec{i}_SKU"] = sku
            output_row[f"Rec{i}_Description"] = desc
            output_row[f"Rec{i}_Expected_Qty"] = expected_qty
            output_row[f"Rec{i}_UnitPrice"] = avg_unit_price
            output_row[f"Rec{i}_Revenue"] = revenue

        output_row["Total_Expected_Revenue"] = round(total_revenue, 2)

        rows.append(output_row)

    revenue_df = pd.DataFrame(rows)

    revenue_df = revenue_df.sort_values(
        by="Total_Expected_Revenue",
        ascending=False,
    )

    return revenue_df

st.title("🛍️ Retail Customer SKU Recommendation System")

st.markdown(
    """
    ### About the Solution

    This AI-powered Retail Customer Recommendation System analyzes historical customer transaction patterns and automatically identifies potential products that customers are likely to purchase next.

    Using Machine Learning-based customer segmentation and recommendation techniques, the solution helps businesses:

    - Improve cross-selling and upselling opportunities
    - Increase customer engagement and repeat purchases
    - Predict potential revenue opportunities from recommended products
    - Identify high-value product combinations based on purchasing behaviour

    The application dynamically generates SKU recommendations and estimated revenue opportunities directly from the uploaded dataset.
    """
)

st.caption(
    "Upload the customer transaction dataset and generate customer-level SKU recommendations with expected revenue opportunity."
)


with st.sidebar:
    st.header("Output Controls")

    number_of_customers = st.slider(
        "Number of Customers to Display",
        min_value=10,
        max_value=100,
        value=10,
        step=10,
    )

    number_of_skus = st.slider(
        "Number of SKU Recommendations",
        min_value=3,
        max_value=50,
        value=3,
        step=1,
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
    prediction_df, cleaned_df = generate_prediction_output(
        raw_df,
        number_of_skus,
    )


prediction_df["CustomerID"] = prediction_df["CustomerID"].astype(str)

required_cols = get_recommendation_columns(number_of_skus)

available_cols = [
    col for col in required_cols
    if col in prediction_df.columns
]

prediction_df = prediction_df[available_cols]


st.subheader(f"Random {number_of_customers} Customer Recommendations")

sample_size = min(number_of_customers, len(prediction_df))

random_customers = prediction_df.sample(
    n=sample_size,
    random_state=None,
)

st.dataframe(
    random_customers,
    use_container_width=True,
    hide_index=True,
)


st.subheader("Expected Revenue Opportunity")

st.caption(
    "Expected revenue is calculated using the historical average quantity and average unit price for each recommended SKU from the uploaded dataset."
)

revenue_table = build_revenue_opportunity_table(
    random_customers,
    cleaned_df,
    number_of_skus,
)

st.dataframe(
    revenue_table,
    use_container_width=True,
    hide_index=True,
)


total_revenue_opportunity = revenue_table["Total_Expected_Revenue"].sum()

st.info(
    f"Total expected revenue from the displayed {sample_size} customers: "
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
