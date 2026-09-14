import os
import pickle
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def run_customer_segmentation():
    """
    Loads sales data from SQLite, computes RFM features, scales them,
    runs K-Means clustering, performs PCA, and saves segment labels back to the DB.
    Also exports the trained model pipeline.
    """
    print("Starting Customer Segmentation Pipeline...")
    
    # 1. Paths configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    db_path = os.path.join(project_root, "data", "sales.db")
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rfm_model.pkl")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Please run etl/pipeline.py first.")
        
    # 2. Load Sales Data
    conn = sqlite3.connect(db_path)
    
    query = """
        SELECT 
            s.customer_id,
            s.total_price,
            s.sale_date
        FROM fact_sales s
    """
    df_sales = pd.read_sql_query(query, conn)
    
    if df_sales.empty:
        raise ValueError("No sales transaction records found in database!")
        
    df_sales['sale_date'] = pd.to_datetime(df_sales['sale_date'])
    
    # 3. Compute RFM metrics
    print("Computing RFM (Recency, Frequency, Monetary) metrics...")
    
    # We define "today" as 1 day after the latest sale date in the dataset
    ref_date = df_sales['sale_date'].max() + pd.Timedelta(days=1)
    
    rfm = df_sales.groupby('customer_id').agg({
        'sale_date': lambda x: (ref_date - x.max()).days, # Recency
        'customer_id': 'count',                          # Frequency
        'total_price': 'sum'                             # Monetary
    }).rename(columns={
        'sale_date': 'recency',
        'customer_id': 'frequency',
        'total_price': 'monetary'
    }).reset_index()
    
    print(f"RFM calculated for {len(rfm)} unique customers.")
    
    # 4. Standardize / Scale metrics
    print("Standardizing RFM features...")
    features = ['recency', 'frequency', 'monetary']
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])
    
    # 5. Run K-Means Clustering (k=4 segments)
    print("Running K-Means clustering (k=4)...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Define Segment names based on average stats of each cluster
    # Let's inspect centroids to name them logically:
    # Champions: Low Recency, High Frequency, High Monetary
    # Loyal Customers: Low/Medium Recency, Medium Frequency, Medium Monetary
    # New/Recent: Low Recency, Low Frequency, Low Monetary
    # At-Risk/Lost: High Recency, Low Frequency, Low Monetary
    
    cluster_means = rfm.groupby('cluster')[features].mean()
    print("Cluster characteristics (raw means):")
    print(cluster_means)
    
    # Logic to dynamically label clusters:
    # Rank clusters by Monetary (spending) and Recency (dormancy)
    monetary_rank = cluster_means['monetary'].argsort() # Sorted ascending
    recency_rank = cluster_means['recency'].argsort()
    
    # Map clusters to profiles
    # Top spending is VIP / Champions
    champions_cluster = monetary_rank.iloc[3]
    # Lowest spending and highest recency (highest inactivity) is At Risk / Lost
    lost_cluster = recency_rank.iloc[3]
    # Low recency (active) but lower spending is New / Promising
    # We find the remaining clusters after champions and lost are excluded
    active_clusters = [c for c in recency_rank.index if c not in (champions_cluster, lost_cluster)]
    # From the remaining, the one with lower frequency/monetary is "New/Recent", and higher is "Loyal"
    if cluster_means.loc[active_clusters[0], 'monetary'] > cluster_means.loc[active_clusters[1], 'monetary']:
        loyal_cluster = active_clusters[0]
        new_cluster = active_clusters[1]
    else:
        loyal_cluster = active_clusters[1]
        new_cluster = active_clusters[0]
        
    segment_map = {
        champions_cluster: "VIP Champions",
        loyal_cluster: "Loyal Customers",
        new_cluster: "New Customers",
        lost_cluster: "At Risk / Hibernating"
    }
    # A real bug lived here once: if two of the four roles ever resolve to the
    # same cluster label, this dict silently collapses to fewer than 4 keys
    # and rfm['cluster'].map() below leaves the unmapped cluster's customers
    # as NaN segment_name (invisible in a value_counts() print, not a crash).
    assert len(segment_map) == cluster_means.shape[0], (
        f"segment_map has {len(segment_map)} entries but there are "
        f"{cluster_means.shape[0]} clusters -- two roles resolved to the same "
        f"cluster label, so a real cluster of customers would go unlabeled."
    )

    rfm['segment_name'] = rfm['cluster'].map(segment_map)
    assert not rfm['segment_name'].isna().any(), "Some customers got no segment_name -- segment_map is missing a cluster."
    print("\nSegment assignments count:")
    print(rfm['segment_name'].value_counts())
    
    # 6. Dimensionality reduction (PCA) for visual plotting
    print("Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=2, random_state=42)
    rfm_pca = pca.fit_transform(rfm_scaled)
    rfm['pca_1'] = rfm_pca[:, 0]
    rfm['pca_2'] = rfm_pca[:, 1]
    
    # 7. Save segment and PCA columns back to the SQLite Database
    print("Updating customer database with segment designations...")
    cursor = conn.cursor()
    
    # Ensure dim_customers table has pca_1, pca_2, and segment columns 
    # (segment is already in schema, let's add pca_1 and pca_2 if not exists)
    try:
        cursor.execute("ALTER TABLE dim_customers ADD COLUMN pca_1 REAL;")
        cursor.execute("ALTER TABLE dim_customers ADD COLUMN pca_2 REAL;")
    except sqlite3.OperationalError:
        # Columns already exist, which is fine
        pass
        
    for idx, row in rfm.iterrows():
        cursor.execute(
            """
            UPDATE dim_customers 
            SET segment = ?, pca_1 = ?, pca_2 = ?
            WHERE customer_id = ?
            """,
            (row['segment_name'], float(row['pca_1']), float(row['pca_2']), int(row['customer_id']))
        )
        
    conn.commit()
    conn.close()
    print("Database dim_customers updated successfully.")
    
    # 8. Save models
    model_data = {
        "scaler": scaler,
        "kmeans": kmeans,
        "pca": pca,
        "segment_map": segment_map,
        "features": features
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"Model and scaler saved to {model_path}")
    
if __name__ == "__main__":
    run_customer_segmentation()
