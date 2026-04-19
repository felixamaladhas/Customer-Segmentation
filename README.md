# Retail Customer Segmentation & Recommendation - Streamlit App

This repository contains a Streamlit deployment package for your **Retail Customer Segmentation & Recommendation System**.

## Files included

- `app.py` - Streamlit frontend
- `retail_segmentation_recommendation_pipeline.py` - full ML pipeline
- `requirements.txt` - Python dependencies for Streamlit Cloud / GitHub deployment
- `runtime.txt` - Python runtime version
- `.gitignore` - files to exclude from Git

## What the app does

- Upload Online Retail dataset (`.xlsx`, `.xls`, `.csv`)
- Clean and preprocess transaction data
- Engineer customer features
- Detect outliers using Isolation Forest
- Apply scaling and PCA
- Run KMeans clustering
- Generate cluster-based product recommendations
- Visualize clusters and silhouette scores
- Download outputs and trained model artifacts

## Recommended GitHub repository structure

```text
your-repo/
├─ app.py
├─ retail_segmentation_recommendation_pipeline.py
├─ requirements.txt
├─ runtime.txt
├─ .gitignore
└─ README.md
```

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud deployment

1. Create a new GitHub repository
2. Upload all files from this folder
3. Go to Streamlit Community Cloud
4. Connect your GitHub repo
5. Set main file path to `app.py`
6. Deploy

## Notes

- The app expects the same Online Retail style columns used in your notebook:
  - `InvoiceNo`
  - `StockCode`
  - `Description`
  - `Quantity`
  - `InvoiceDate`
  - `UnitPrice`
  - `CustomerID`
  - `Country`
- If you use the original UCI Online Retail dataset, the app should work directly.
