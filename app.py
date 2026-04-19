from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from retail_segmentation_recommendation_pipeline import (
    build_customer_features,
    choose_best_k,
    clean_transactions,
    cluster_customers,
    detect_outliers,
    generate_recommendations,
    load_data,
    run_pipeline,
    scale_features,
    apply_pca,
)


st.set_page_config(
    page_title="Retail Customer Segmentation & Recommendation",
    page_icon="🛍️",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_uploaded_dataset(file_bytes: bytes, suffix: str) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name
    return load_data(temp_path)


@st.cache_data(show_spinner=False)
def prepare_quick_preview(df: pd.DataFrame):
    cleaned = clean_transactions(df)
    customer_data = build_customer_features(cleaned)
    customer_data_cleaned, outliers = detect_outliers(customer_data, contamination=0.05)
    customer_data_scaled, _, _ = scale_features(customer_data_cleaned)
    customer_data_pca, _ = apply_pca(customer_data_scaled, n_components=3)
    best_k, silhouette_scores = choose_best_k(customer_data_pca, start_k=3, stop_k=8)
    customer_clusters, pca_clusters, _ = cluster_customers(
        customer_data_cleaned, customer_data_pca, n_clusters=best_k
    )
    customer_recommendations, top_products_per_cluster, recommendation_dict = generate_recommendations(
        cleaned,
        customer_clusters,
        outliers,
        top_n_products_per_cluster=10,
        n_recommendations=3,
    )
    return {
        "cleaned": cleaned,
        "customer_data": customer_data,
        "customer_data_cleaned": customer_data_cleaned,
        "outliers": outliers,
        "customer_data_pca": customer_data_pca,
        "pca_clusters": pca_clusters,
        "customer_clusters": customer_clusters,
        "customer_recommendations": customer_recommendations,
        "top_products_per_cluster": top_products_per_cluster,
        "recommendation_dict": recommendation_dict,
        "best_k": best_k,
        "silhouette_scores": silhouette_scores,
    }


def plot_cluster_distribution(customer_clusters: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    cluster_counts = customer_clusters["cluster"].value_counts().sort_index()
    ax.bar(cluster_counts.index.astype(str), cluster_counts.values)
    ax.set_title("Customer Count by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Customers")
    st.pyplot(fig)



def plot_pca_scatter(pca_clusters: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        pca_clusters["PC1"],
        pca_clusters["PC2"],
        c=pca_clusters["cluster"],
    )
    ax.set_title("Customer Segments in PCA Space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    st.pyplot(fig)



def plot_silhouette_scores(scores: dict[int, float]):
    fig, ax = plt.subplots(figsize=(8, 4))
    ks = list(scores.keys())
    vals = list(scores.values())
    ax.plot(ks, vals, marker="o")
    ax.set_title("Silhouette Score by k")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


st.title("🛍️ Retail Customer Segmentation & Recommendation System")
st.caption("Upload your Online Retail dataset, generate customer segments, and explore product recommendations.")

with st.sidebar:
    st.header("Configuration")
    pca_components = st.slider("PCA components", min_value=2, max_value=6, value=3)
    force_k = st.selectbox("Cluster selection", options=["Auto", 3, 4, 5, 6, 7, 8], index=0)
    top_n_products = st.slider("Top products per cluster", min_value=5, max_value=20, value=10)
    n_recommendations = st.slider("Recommendations per customer", min_value=1, max_value=5, value=3)

uploaded_file = st.file_uploader(
    "Upload Online Retail dataset (.xlsx, .xls, .csv)",
    type=["xlsx", "xls", "csv"],
)

if uploaded_file is None:
    st.info("Upload your dataset to start.")
    st.stop()

suffix = Path(uploaded_file.name).suffix.lower()
raw_df = load_uploaded_dataset(uploaded_file.getvalue(), suffix)

st.subheader("Dataset Preview")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{len(raw_df):,}")
col2.metric("Columns", len(raw_df.columns))
col3.metric("Unique customers", f"{raw_df['CustomerID'].nunique() if 'CustomerID' in raw_df.columns else 0:,}")
st.dataframe(raw_df.head(20), use_container_width=True)

preview = prepare_quick_preview(raw_df)

st.subheader("Project Summary")
met1, met2, met3, met4 = st.columns(4)
met1.metric("Cleaned transactions", f"{len(preview['cleaned']):,}")
met2.metric("Customers modeled", f"{len(preview['customer_data_cleaned']):,}")
met3.metric("Outlier customers", f"{len(preview['outliers']):,}")
met4.metric("Suggested best k", preview["best_k"])


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Cluster Insights",
    "Recommendations",
    "Customer Lookup",
    "Downloads",
    "Run Full Pipeline",
])

with tab1:
    st.markdown("### Cluster Distribution")
    plot_cluster_distribution(preview["customer_clusters"])

    st.markdown("### PCA View")
    plot_pca_scatter(preview["pca_clusters"])

    st.markdown("### Silhouette Analysis")
    plot_silhouette_scores(preview["silhouette_scores"])

    st.markdown("### Top Products per Cluster")
    st.dataframe(preview["top_products_per_cluster"], use_container_width=True)

with tab2:
    st.markdown("### Customer Recommendations")
    st.dataframe(preview["customer_recommendations"].head(200), use_container_width=True)

with tab3:
    st.markdown("### Search by Customer ID")
    customer_id = st.text_input("Enter CustomerID")
    if customer_id:
        customer_id = str(customer_id).strip()
        customer_rows = preview["customer_recommendations"][preview["customer_recommendations"]["CustomerID"] == customer_id]
        if customer_rows.empty:
            st.warning("CustomerID not found in the modeled dataset.")
        else:
            st.dataframe(customer_rows, use_container_width=True)

with tab4:
    st.markdown("### Download Preview Outputs")
    st.download_button(
        "Download customer recommendations CSV",
        data=preview["customer_recommendations"].to_csv(index=False).encode("utf-8"),
        file_name="customer_recommendations.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download top products per cluster CSV",
        data=preview["top_products_per_cluster"].to_csv(index=False).encode("utf-8"),
        file_name="top_products_per_cluster.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download silhouette scores JSON",
        data=json.dumps(preview["silhouette_scores"], indent=2).encode("utf-8"),
        file_name="silhouette_scores.json",
        mime="application/json",
    )

with tab5:
    st.markdown("### Run Full Training Pipeline and Save Artifacts")
    st.write("This creates the trained model files required for reuse or deployment.")

    if st.button("Run full pipeline"):
        with st.spinner("Running pipeline..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = Path(temp_dir) / f"input{suffix}"
                input_path.write_bytes(uploaded_file.getvalue())
                output_dir = Path(temp_dir) / "outputs"

                clusters, pca_clusters, recommendations, artifacts = run_pipeline(
                    input_path=input_path,
                    output_dir=output_dir,
                    n_pca_components=int(pca_components),
                    start_k=3,
                    stop_k=8,
                    force_k=None if force_k == "Auto" else int(force_k),
                )

                metadata = {
                    "best_k": artifacts.best_k,
                    "silhouette_scores": artifacts.silhouette_scores,
                    "feature_columns": artifacts.feature_columns,
                }

                st.success("Pipeline completed successfully.")
                st.write(f"Selected/used k: {artifacts.best_k}")
                st.dataframe(recommendations.head(100), use_container_width=True)

                for file_name in [
                    "customer_clusters.csv",
                    "customer_pca_clusters.csv",
                    "customer_outliers.csv",
                    "customer_recommendations.csv",
                    "top_products_per_cluster.csv",
                    "model_metadata.json",
                    "scaler.pkl",
                    "pca.pkl",
                    "kmeans.pkl",
                    "feature_columns.pkl",
                    "cluster_recommendations.pkl",
                ]:
                    path = output_dir / file_name
                    if path.exists():
                        mime = "application/octet-stream"
                        if file_name.endswith(".csv"):
                            mime = "text/csv"
                        elif file_name.endswith(".json"):
                            mime = "application/json"
                        st.download_button(
                            f"Download {file_name}",
                            data=path.read_bytes(),
                            file_name=file_name,
                            mime=mime,
                            key=f"download_{file_name}",
                        )
