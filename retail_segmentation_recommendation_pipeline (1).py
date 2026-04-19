"""
End-to-end Retail Customer Segmentation & Recommendation pipeline.

Based on the attached notebook:
Enhancing_Retail_Marketing_through_Customer_Segmentation_and_ML_based_Recommendations.ipynb

What this script does:
1. Loads the Online Retail Excel dataset
2. Cleans transactions
3. Engineers customer-level features
4. Detects outlier customers with Isolation Forest
5. Scales features and applies PCA
6. Clusters customers with KMeans
7. Generates top-product recommendations per cluster
8. Saves outputs and trained objects

Usage:
    python retail_segmentation_recommendation_pipeline.py --input /path/to/OnlineRetail.xlsx --output_dir outputs
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


@dataclass
class PipelineArtifacts:
    scaler: StandardScaler
    pca: PCA
    kmeans: KMeans
    feature_columns: List[str]
    top_products_per_cluster: pd.DataFrame
    recommendation_dict: Dict[str, Dict[str, str]]
    best_k: int
    silhouette_scores: Dict[int, float]


# -----------------------------
# Data loading and cleaning
# -----------------------------

def load_data(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    elif file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path, encoding="latin1")
    else:
        raise ValueError("Supported input formats are .xlsx, .xls, and .csv")

    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
        "UnitPrice", "CustomerID", "Country"
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.copy()

    # Remove rows missing key business identifiers
    df = df.dropna(subset=["CustomerID", "Description"])

    # Remove exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Mark cancellations
    df["Transaction_Status"] = np.where(
        df["InvoiceNo"].astype(str).str.startswith("C"),
        "Cancelled",
        "Completed",
    )

    # Remove anomalous stock codes with too few numeric chars
    unique_codes = pd.Series(df["StockCode"].astype(str).unique())
    bad_codes = unique_codes[
        unique_codes.apply(lambda x: sum(c.isdigit() for c in str(x))) < 2
    ].tolist()
    df = df[~df["StockCode"].astype(str).isin(bad_codes)]

    # Remove service / admin type rows seen in the notebook
    service_related_descriptions = ["Next Day Carriage", "High Resolution Image"]
    df = df[~df["Description"].isin(service_related_descriptions)]

    # Remove records with non-positive price
    df = df[df["UnitPrice"] > 0]

    # Keep positive quantity rows for recommendation and spend logic
    # Cancelled rows often contain negative quantities; keep status info but exclude from modeling features
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    return df.reset_index(drop=True)


# -----------------------------
# Feature engineering
# -----------------------------

def calculate_trend(spend_data: pd.Series) -> float:
    values = spend_data.to_numpy()
    if len(values) > 1:
        x = np.arange(len(values))
        slope, _, _, _, _ = linregress(x, values)
        return float(slope)
    return 0.0


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["InvoiceDay"] = data["InvoiceDate"].dt.normalize()
    data["Total_Spend"] = data["UnitPrice"] * data["Quantity"]

    # Use only positive-quantity rows for most customer-behaviour features
    purchase_df = data[data["Quantity"] > 0].copy()

    most_recent_date = purchase_df["InvoiceDay"].max()

    customer_data = purchase_df.groupby("CustomerID")["InvoiceDay"].max().reset_index()
    customer_data["Days_Since_Last_Purchase"] = (
        most_recent_date - customer_data["InvoiceDay"]
    ).dt.days
    customer_data = customer_data.drop(columns=["InvoiceDay"])

    total_transactions = purchase_df.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
    total_transactions = total_transactions.rename(columns={"InvoiceNo": "Total_Transactions"})

    total_products_purchased = purchase_df.groupby("CustomerID")["Quantity"].sum().reset_index()
    total_products_purchased = total_products_purchased.rename(columns={"Quantity": "Total_Products_Purchased"})

    total_spend = purchase_df.groupby("CustomerID")["Total_Spend"].sum().reset_index()

    average_transaction_value = total_spend.merge(total_transactions, on="CustomerID")
    average_transaction_value["Average_Transaction_Value"] = (
        average_transaction_value["Total_Spend"] / average_transaction_value["Total_Transactions"]
    )
    average_transaction_value = average_transaction_value[["CustomerID", "Average_Transaction_Value"]]

    unique_products_purchased = purchase_df.groupby("CustomerID")["StockCode"].nunique().reset_index()
    unique_products_purchased = unique_products_purchased.rename(
        columns={"StockCode": "Unique_Products_Purchased"}
    )

    purchase_df["Day_Of_Week"] = purchase_df["InvoiceDate"].dt.dayofweek
    purchase_df["Hour"] = purchase_df["InvoiceDate"].dt.hour

    days_between_purchases = purchase_df.sort_values(["CustomerID", "InvoiceDate"]).groupby("CustomerID")["InvoiceDay"].apply(
        lambda x: x.diff().dropna().dt.days
    )
    average_days_between_purchases = days_between_purchases.groupby("CustomerID").mean().reset_index()
    average_days_between_purchases = average_days_between_purchases.rename(
        columns={"InvoiceDay": "Average_Days_Between_Purchases", 0: "Average_Days_Between_Purchases"}
    )
    if "Average_Days_Between_Purchases" not in average_days_between_purchases.columns:
        average_days_between_purchases.columns = ["CustomerID", "Average_Days_Between_Purchases"]

    favorite_shopping_day = (
        purchase_df.groupby(["CustomerID", "Day_Of_Week"]).size().reset_index(name="Count")
    )
    favorite_shopping_day = favorite_shopping_day.loc[
        favorite_shopping_day.groupby("CustomerID")["Count"].idxmax(),
        ["CustomerID", "Day_Of_Week"],
    ]

    favorite_shopping_hour = (
        purchase_df.groupby(["CustomerID", "Hour"]).size().reset_index(name="Count")
    )
    favorite_shopping_hour = favorite_shopping_hour.loc[
        favorite_shopping_hour.groupby("CustomerID")["Count"].idxmax(),
        ["CustomerID", "Hour"],
    ]

    customer_country = purchase_df.groupby(["CustomerID", "Country"]).size().reset_index(name="Number_of_Transactions")
    customer_main_country = customer_country.sort_values("Number_of_Transactions", ascending=False).drop_duplicates("CustomerID")
    customer_main_country["Is_UK"] = (customer_main_country["Country"] == "United Kingdom").astype(int)
    customer_main_country = customer_main_country[["CustomerID", "Is_UK"]]

    cancelled_transactions = data[data["Transaction_Status"] == "Cancelled"]
    cancellation_frequency = cancelled_transactions.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
    cancellation_frequency = cancellation_frequency.rename(columns={"InvoiceNo": "Cancellation_Frequency"})

    purchase_df["Year"] = purchase_df["InvoiceDate"].dt.year
    purchase_df["Month"] = purchase_df["InvoiceDate"].dt.month

    monthly_spending = purchase_df.groupby(["CustomerID", "Year", "Month"])["Total_Spend"].sum().reset_index()

    seasonal_buying_patterns = monthly_spending.groupby("CustomerID")["Total_Spend"].agg(["mean", "std"]).reset_index()
    seasonal_buying_patterns = seasonal_buying_patterns.rename(
        columns={"mean": "Monthly_Spending_Mean", "std": "Monthly_Spending_Std"}
    )
    seasonal_buying_patterns["Monthly_Spending_Std"] = seasonal_buying_patterns["Monthly_Spending_Std"].fillna(0)

    spending_trends = monthly_spending.groupby("CustomerID")["Total_Spend"].apply(calculate_trend).reset_index()
    spending_trends = spending_trends.rename(columns={"Total_Spend": "Spending_Trend"})

    customer_data = customer_data.merge(total_transactions, on="CustomerID", how="left")
    customer_data = customer_data.merge(total_products_purchased, on="CustomerID", how="left")
    customer_data = customer_data.merge(total_spend, on="CustomerID", how="left")
    customer_data = customer_data.merge(average_transaction_value, on="CustomerID", how="left")
    customer_data = customer_data.merge(unique_products_purchased, on="CustomerID", how="left")
    customer_data = customer_data.merge(average_days_between_purchases, on="CustomerID", how="left")
    customer_data = customer_data.merge(favorite_shopping_day, on="CustomerID", how="left")
    customer_data = customer_data.merge(favorite_shopping_hour, on="CustomerID", how="left")
    customer_data = customer_data.merge(customer_main_country, on="CustomerID", how="left")
    customer_data = customer_data.merge(cancellation_frequency, on="CustomerID", how="left")
    customer_data = customer_data.merge(seasonal_buying_patterns, on="CustomerID", how="left")
    customer_data = customer_data.merge(spending_trends, on="CustomerID", how="left")

    customer_data["Cancellation_Frequency"] = customer_data["Cancellation_Frequency"].fillna(0)
    customer_data["Cancellation_Rate"] = (
        customer_data["Cancellation_Frequency"] / customer_data["Total_Transactions"].replace(0, np.nan)
    ).fillna(0)

    customer_data["Average_Days_Between_Purchases"] = customer_data["Average_Days_Between_Purchases"].fillna(0)
    customer_data["Is_UK"] = customer_data["Is_UK"].fillna(0).astype(int)
    customer_data["Day_Of_Week"] = customer_data["Day_Of_Week"].fillna(0).astype(int)
    customer_data["Hour"] = customer_data["Hour"].fillna(0).astype(int)

    customer_data["CustomerID"] = customer_data["CustomerID"].astype(str)

    return customer_data.sort_values("CustomerID").reset_index(drop=True)


# -----------------------------
# Outlier detection, scaling, PCA, clustering
# -----------------------------

def detect_outliers(customer_data: pd.DataFrame, contamination: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = customer_data.copy()
    model = IsolationForest(contamination=contamination, random_state=0)

    features = data.drop(columns=["CustomerID"])
    preds = model.fit_predict(features.to_numpy())

    data["Outlier_Scores"] = preds
    data["Is_Outlier"] = np.where(data["Outlier_Scores"] == -1, 1, 0)

    cleaned = data[data["Is_Outlier"] == 0].drop(columns=["Outlier_Scores", "Is_Outlier"]).reset_index(drop=True)
    outliers = data[data["Is_Outlier"] == 1].reset_index(drop=True)
    return cleaned, outliers


def scale_features(customer_data_cleaned: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
    data = customer_data_cleaned.copy()

    columns_to_exclude = ["CustomerID", "Is_UK", "Day_Of_Week"]
    feature_columns = [c for c in data.columns if c not in columns_to_exclude]

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(data[feature_columns])
    scaled_df = pd.DataFrame(scaled_array, columns=feature_columns, index=data.index)

    # Add back excluded columns unscaled to stay close to notebook logic
    scaled_df["Is_UK"] = data["Is_UK"].values
    scaled_df["Day_Of_Week"] = data["Day_Of_Week"].values
    scaled_df["CustomerID"] = data["CustomerID"].values

    ordered_columns = [c for c in data.columns if c != "CustomerID"]
    scaled_df = scaled_df[ordered_columns + ["CustomerID"]]
    scaled_df = scaled_df.set_index("CustomerID")

    return scaled_df, scaler, feature_columns


def apply_pca(customer_data_scaled: pd.DataFrame, n_components: int = 6) -> Tuple[pd.DataFrame, PCA]:
    pca = PCA(n_components=n_components, random_state=0)
    transformed = pca.fit_transform(customer_data_scaled)
    pca_df = pd.DataFrame(
        transformed,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=customer_data_scaled.index,
    )
    return pca_df, pca


def choose_best_k(pca_df: pd.DataFrame, start_k: int = 3, stop_k: int = 10) -> Tuple[int, Dict[int, float]]:
    scores: Dict[int, float] = {}
    for k in range(start_k, stop_k + 1):
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=100, random_state=0)
        labels = model.fit_predict(pca_df)
        scores[k] = silhouette_score(pca_df, labels)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def cluster_customers(
    customer_data_cleaned: pd.DataFrame,
    customer_data_pca: pd.DataFrame,
    n_clusters: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, KMeans]:
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, max_iter=100, random_state=0)
    labels = kmeans.fit_predict(customer_data_pca)

    # Reorder cluster labels by cluster size for consistent interpretation
    cluster_sizes = pd.Series(labels).value_counts().sort_values(ascending=False)
    remap = {old: new for new, old in enumerate(cluster_sizes.index)}
    remapped_labels = np.array([remap[label] for label in labels])

    customer_data_cleaned = customer_data_cleaned.copy()
    customer_data_pca = customer_data_pca.copy()
    customer_data_cleaned["cluster"] = remapped_labels
    customer_data_pca["cluster"] = remapped_labels

    return customer_data_cleaned, customer_data_pca, kmeans


# -----------------------------
# Recommendation engine
# -----------------------------

def generate_recommendations(
    original_df: pd.DataFrame,
    customer_data_cleaned: pd.DataFrame,
    outliers: pd.DataFrame,
    top_n_products_per_cluster: int = 10,
    n_recommendations: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, str]]]:
    df = original_df.copy()

    outlier_customer_ids = set(outliers["CustomerID"].astype(str).tolist())
    df_filtered = df[df["CustomerID"].astype(str).isin(set(customer_data_cleaned["CustomerID"]))]
    df_filtered = df_filtered[~df_filtered["CustomerID"].astype(str).isin(outlier_customer_ids)]
    df_filtered = df_filtered[df_filtered["Quantity"] > 0].copy()
    df_filtered["CustomerID"] = df_filtered["CustomerID"].astype(str)

    cluster_map = customer_data_cleaned[["CustomerID", "cluster"]].copy()
    merged_data = df_filtered.merge(cluster_map, on="CustomerID", how="inner")

    best_selling_products = (
        merged_data.groupby(["cluster", "StockCode", "Description"])["Quantity"]
        .sum()
        .reset_index()
        .sort_values(["cluster", "Quantity"], ascending=[True, False])
    )
    top_products_per_cluster = best_selling_products.groupby("cluster").head(top_n_products_per_cluster).reset_index(drop=True)

    customer_purchases = (
        merged_data.groupby(["CustomerID", "cluster", "StockCode"])["Quantity"]
        .sum()
        .reset_index()
    )

    recommendations = []
    recommendation_dict: Dict[str, Dict[str, str]] = {}

    for cluster in sorted(top_products_per_cluster["cluster"].unique()):
        top_products = top_products_per_cluster[top_products_per_cluster["cluster"] == cluster]
        customers_in_cluster = customer_data_cleaned.loc[
            customer_data_cleaned["cluster"] == cluster, "CustomerID"
        ].tolist()

        for customer in customers_in_cluster:
            purchased = customer_purchases.loc[
                (customer_purchases["CustomerID"] == customer) &
                (customer_purchases["cluster"] == cluster),
                "StockCode",
            ].astype(str).tolist()

            top_products_not_purchased = top_products[~top_products["StockCode"].astype(str).isin(purchased)].head(n_recommendations)

            row = {
                "CustomerID": customer,
                "cluster": cluster,
            }
            customer_recs: Dict[str, str] = {}

            for idx in range(n_recommendations):
                rec_no = idx + 1
                if idx < len(top_products_not_purchased):
                    row[f"Rec{rec_no}_StockCode"] = str(top_products_not_purchased.iloc[idx]["StockCode"])
                    row[f"Rec{rec_no}_Description"] = str(top_products_not_purchased.iloc[idx]["Description"])
                    customer_recs[f"Rec{rec_no}"] = str(top_products_not_purchased.iloc[idx]["Description"])
                else:
                    row[f"Rec{rec_no}_StockCode"] = None
                    row[f"Rec{rec_no}_Description"] = None
                    customer_recs[f"Rec{rec_no}"] = None

            recommendations.append(row)
            recommendation_dict[customer] = customer_recs

    recommendations_df = pd.DataFrame(recommendations)
    final_df = customer_data_cleaned.merge(recommendations_df, on=["CustomerID", "cluster"], how="left")

    return final_df, top_products_per_cluster, recommendation_dict


# -----------------------------
# Save outputs
# -----------------------------

def save_artifacts(output_dir: str | Path, artifacts: PipelineArtifacts) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(artifacts.scaler, f)

    with open(output_dir / "pca.pkl", "wb") as f:
        pickle.dump(artifacts.pca, f)

    with open(output_dir / "kmeans.pkl", "wb") as f:
        pickle.dump(artifacts.kmeans, f)

    with open(output_dir / "feature_columns.pkl", "wb") as f:
        pickle.dump(artifacts.feature_columns, f)

    with open(output_dir / "cluster_recommendations.pkl", "wb") as f:
        pickle.dump(artifacts.recommendation_dict, f)

    artifacts.top_products_per_cluster.to_csv(output_dir / "top_products_per_cluster.csv", index=False)

    metadata = {
        "best_k": artifacts.best_k,
        "silhouette_scores": artifacts.silhouette_scores,
        "feature_columns": artifacts.feature_columns,
    }
    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    n_pca_components: int = 6,
    start_k: int = 3,
    stop_k: int = 10,
    force_k: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PipelineArtifacts]:
    print("Loading data...")
    raw_df = load_data(input_path)

    print("Cleaning transactions...")
    clean_df = clean_transactions(raw_df)

    print("Building customer features...")
    customer_data = build_customer_features(clean_df)

    print("Detecting outliers...")
    customer_data_cleaned, outliers = detect_outliers(customer_data, contamination=0.05)

    print("Scaling features...")
    customer_data_scaled, scaler, feature_columns = scale_features(customer_data_cleaned)

    print("Applying PCA...")
    customer_data_pca, pca = apply_pca(customer_data_scaled, n_components=n_pca_components)

    print("Selecting k using silhouette score...")
    best_k, silhouette_scores = choose_best_k(customer_data_pca, start_k=start_k, stop_k=stop_k)
    chosen_k = force_k if force_k is not None else best_k
    print(f"Best k from silhouette analysis: {best_k}")
    print(f"Using k = {chosen_k}")

    print("Clustering customers...")
    customer_clusters, pca_clusters, kmeans = cluster_customers(customer_data_cleaned, customer_data_pca, n_clusters=chosen_k)

    print("Generating recommendations...")
    customer_recommendations, top_products_per_cluster, recommendation_dict = generate_recommendations(
        clean_df,
        customer_clusters,
        outliers,
        top_n_products_per_cluster=10,
        n_recommendations=3,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    customer_clusters.to_csv(output_dir / "customer_clusters.csv", index=False)
    pca_clusters.to_csv(output_dir / "customer_pca_clusters.csv")
    outliers.to_csv(output_dir / "customer_outliers.csv", index=False)
    customer_recommendations.to_csv(output_dir / "customer_recommendations.csv", index=False)

    artifacts = PipelineArtifacts(
        scaler=scaler,
        pca=pca,
        kmeans=kmeans,
        feature_columns=feature_columns,
        top_products_per_cluster=top_products_per_cluster,
        recommendation_dict=recommendation_dict,
        best_k=chosen_k,
        silhouette_scores=silhouette_scores,
    )
    save_artifacts(output_dir, artifacts)

    print("Done.")
    print(f"Outputs saved to: {output_dir.resolve()}")

    return customer_clusters, pca_clusters, customer_recommendations, artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retail customer segmentation and recommendation pipeline")
    parser.add_argument("--input", required=True, help="Path to input Excel/CSV dataset")
    parser.add_argument("--output_dir", default="outputs", help="Directory to save outputs")
    parser.add_argument("--pca_components", type=int, default=6, help="Number of PCA components")
    parser.add_argument("--start_k", type=int, default=3, help="Minimum k for silhouette search")
    parser.add_argument("--stop_k", type=int, default=10, help="Maximum k for silhouette search")
    parser.add_argument("--force_k", type=int, default=None, help="Force a fixed cluster count instead of best silhouette k")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        n_pca_components=args.pca_components,
        start_k=args.start_k,
        stop_k=args.stop_k,
        force_k=args.force_k,
    )
