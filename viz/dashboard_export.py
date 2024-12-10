import os
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_dashboard():
    """
    Generates a high-quality, dark-mode Matplotlib/Seaborn dashboard image.
    Contains KPI summaries, PCA Customer Clusters, Monthly Revenue, and Category Sales.
    """
    print("Generating Power BI style Sales Dashboard visual...")
    
    # 1. Path configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    db_path = os.path.join(project_root, "data", "sales.db")
    viz_dir = os.path.join(project_root, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    export_path = os.path.join(viz_dir, "powerbi_sales_dashboard.png")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Please run etl/pipeline.py and models/segmentation.py first.")
        
    # 2. Extract Data from SQLite
    conn = sqlite3.connect(db_path)
    
    # KPI metrics
    kpis = pd.read_sql_query("""
        SELECT 
            SUM(total_price) as total_revenue,
            COUNT(sale_id) as total_orders,
            SUM(units_sold) as total_units,
            (SELECT COUNT(*) FROM dim_customers) as total_customers
        FROM fact_sales
    """, conn).iloc[0]
    
    # Customers PCA data
    df_customers = pd.read_sql_query("""
        SELECT customer_id, customer_name, segment, pca_1, pca_2
        FROM dim_customers
    """, conn)
    
    # Monthly sales trend
    df_monthly = pd.read_sql_query("""
        SELECT 
            strftime('%Y-%m', sale_date) as month,
            SUM(total_price) as revenue,
            SUM(units_sold) as units
        FROM fact_sales
        GROUP BY month
        ORDER BY month
    """, conn)
    
    # Category sales distribution
    df_categories = pd.read_sql_query("""
        SELECT 
            p.category,
            SUM(s.total_price) as revenue
        FROM fact_sales s
        JOIN dim_products p ON s.product_id = p.product_id
        GROUP BY p.category
        ORDER BY revenue DESC
    """, conn)
    # Export tables to CSV for easy import into Power BI
    pd.read_sql_query("SELECT * FROM dim_customers", conn).to_csv(os.path.join(project_root, "data", "dim_customers.csv"), index=False)
    pd.read_sql_query("SELECT * FROM dim_products", conn).to_csv(os.path.join(project_root, "data", "dim_products.csv"), index=False)
    pd.read_sql_query("SELECT * FROM fact_sales", conn).to_csv(os.path.join(project_root, "data", "fact_sales.csv"), index=False)
    print("CSV files exported to data/ directory for Power BI.")

    conn.close()
    
    # 3. Setup styling details
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['text.color'] = '#f8fafc'
    plt.rcParams['axes.labelcolor'] = '#94a3b8'
    plt.rcParams['xtick.color'] = '#94a3b8'
    plt.rcParams['ytick.color'] = '#94a3b8'
    plt.rcParams['grid.color'] = '#334155'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.5
    
    # Create figure
    fig = plt.figure(figsize=(18, 12), facecolor='#0f172a')
    
    # Gridspec layout
    # 4 rows: Row 0 is title, Row 1 is KPIs, Row 2 is PCA Scatter, Row 3 is Line & Bar charts
    gs = fig.add_gridspec(4, 4, height_ratios=[0.08, 0.12, 0.40, 0.40], hspace=0.35, wspace=0.3)
    
    # A. Dashboard Title
    title_ax = fig.add_subplot(gs[0, :], facecolor='none')
    title_ax.axis('off')
    title_ax.text(0.0, 0.5, "EXECUTIVE SALES & CUSTOMER SEGMENTATION DASHBOARD", 
                  fontsize=22, fontweight='bold', color='#38bdf8', va='center')
    title_ax.text(1.0, 0.5, f"Data Refreshed: {datetime.now().strftime('%Y-%m-%d')} | Data Period: 180 Days", 
                  fontsize=10, color='#64748b', ha='right', va='center')
    
    # B. KPI Cards (4 subplots in row 1)
    kpi_labels = ["Total Revenue", "Total Customers", "Total Transactions", "Avg Order Value (AOV)"]
    kpi_values = [
        f"${kpis['total_revenue']:,.2f}",
        f"{kpis['total_customers']:,}",
        f"{kpis['total_orders']:,}",
        f"${kpis['total_revenue'] / kpis['total_orders']:.2f}"
    ]
    kpi_colors = ["#10b981", "#38bdf8", "#f43f5e", "#fbbf24"]
    
    for i in range(4):
        ax = fig.add_subplot(gs[1, i], facecolor='#1e293b')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_color(kpi_colors[i])
        ax.spines['left'].set_linewidth(4)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Text positioning
        ax.text(0.1, 0.7, kpi_labels[i], fontsize=11, color='#94a3b8', fontweight='semibold')
        ax.text(0.1, 0.25, kpi_values[i], fontsize=20, color='#f8fafc', fontweight='bold')
        
    # C. Plot 1: RFM Customer Segments (PCA Projection)
    ax_pca = fig.add_subplot(gs[2, :], facecolor='#1e293b')
    ax_pca.set_title("Customer Segmentation Map (PCA 2D Cluster Projection)", fontsize=14, fontweight='bold', pad=15, loc='left', color='#f8fafc')
    
    # Palette mapping
    palette = {
        "VIP Champions": "#10b981",
        "Loyal Customers": "#38bdf8",
        "New Customers": "#fbbf24",
        "At Risk / Hibernating": "#ef4444"
    }
    
    # Fallback palette for safety
    unique_segments = df_customers['segment'].unique()
    sns_palette = {seg: palette.get(seg, "#a855f7") for seg in unique_segments}
    
    sns.scatterplot(
        data=df_customers,
        x='pca_1',
        y='pca_2',
        hue='segment',
        palette=sns_palette,
        alpha=0.85,
        s=80,
        ax=ax_pca,
        edgecolor='#0f172a',
        linewidth=0.5
    )
    
    ax_pca.set_xlabel("Principal Component 1 (Monetary/Frequency Volume)", fontsize=11, labelpad=10)
    ax_pca.set_ylabel("Principal Component 2 (Recency Inactivity)", fontsize=11, labelpad=10)
    ax_pca.grid(True)
    ax_pca.legend(title="Customer Persona", title_fontsize='11', loc='upper right', facecolor='#1e293b', edgecolor='#334155')
    
    # D. Plot 2: Monthly Revenue Growth Trend
    ax_trend = fig.add_subplot(gs[3, 0:2], facecolor='#1e293b')
    ax_trend.set_title("Monthly Revenue Performance & Growth", fontsize=14, fontweight='bold', pad=15, loc='left', color='#f8fafc')
    
    # Convert month codes to nice labels (e.g., Jun 2025)
    months_dt = pd.to_datetime(df_monthly['month'], format='%Y-%m')
    month_names = months_dt.dt.strftime('%b %Y')
    
    ax_trend.plot(month_names, df_monthly['revenue'], marker='o', linewidth=3, color='#38bdf8', label="Monthly Revenue")
    ax_trend.fill_between(month_names, df_monthly['revenue'], alpha=0.15, color='#38bdf8')
    
    # Adding data labels on line chart
    for i, val in enumerate(df_monthly['revenue']):
        ax_trend.annotate(f"${val/1000:.1f}k", 
                          (month_names[i], val), 
                          textcoords="offset points", 
                          xytext=(0,10), 
                          ha='center', 
                          fontsize=9,
                          fontweight='semibold',
                          color='#cbd5e1')
                          
    ax_trend.set_ylabel("Revenue ($)", fontsize=11, labelpad=10)
    ax_trend.grid(True)
    ax_trend.set_ylim(0, df_monthly['revenue'].max() * 1.25)
    
    # E. Plot 3: Category Sales Distribution (Horizontal Bar Chart)
    ax_cat = fig.add_subplot(gs[3, 2:4], facecolor='#1e293b')
    ax_cat.set_title("Revenue Contribution by Product Category", fontsize=14, fontweight='bold', pad=15, loc='left', color='#f8fafc')
    
    cat_colors = ["#38bdf8", "#10b981", "#fbbf24", "#f43f5e"]
    # Ensure color list is long enough
    while len(cat_colors) < len(df_categories):
        cat_colors.append("#a855f7")
        
    bars = ax_cat.barh(df_categories['category'], df_categories['revenue'], color=cat_colors[:len(df_categories)], height=0.6, edgecolor='#0f172a')
    
    # Adding value labels on horizontal bars
    for bar in bars:
        width = bar.get_width()
        ax_cat.text(width + (df_categories['revenue'].max() * 0.02), 
                    bar.get_y() + bar.get_height()/2, 
                    f"${width:,.2f}", 
                    ha='left', 
                    va='center', 
                    fontsize=10, 
                    fontweight='bold', 
                    color='#f8fafc')
                    
    ax_cat.set_xlabel("Revenue ($)", fontsize=11, labelpad=10)
    ax_cat.grid(True, axis='x')
    ax_cat.set_xlim(0, df_categories['revenue'].max() * 1.15)
    ax_cat.invert_yaxis() # Display top category at the top
    
    # 4. Clean border spines
    for ax in [ax_pca, ax_trend, ax_cat]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#475569')
        ax.spines['bottom'].set_color('#475569')
        
    # Apply tight layout and save
    plt.tight_layout()
    plt.savefig(export_path, dpi=120, facecolor='#0f172a', bbox_inches='tight')
    plt.close()
    
    print(f"Visual dashboard exported successfully to {export_path}")
    
if __name__ == "__main__":
    generate_dashboard()
