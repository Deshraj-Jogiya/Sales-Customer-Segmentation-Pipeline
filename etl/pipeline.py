import os
import sqlite3
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_synthetic_raw_data(num_customers=300, num_products=20, days=180, end_date_str="2025-12-01"):
    """
    Generates realistic sales transaction data using Pareto distribution (80/20 rule).
    Also introduces some missing/dirty data to demonstrate ETL cleaning capabilities.
    """
    print("Generating raw synthetic data...")
    
    # 1. Generate Customers
    customer_names = [
        "James Smith", "Michael Brown", "Robert Jones", "Maria Garcia", "David Miller", 
        "William Davis", "Mary Rodriguez", "Linda Martinez", "Barbara Hernandez", "Elizabeth Lopez",
        "Richard Gonzalez", "Joseph Wilson", "Thomas Anderson", "Patricia Thomas", "Jennifer Taylor",
        "Charles Moore", "Christopher Jackson", "Daniel Martin", "Matthew Lee", "Margaret Perez",
        "Anthony Thompson", "Mark White", "Donald Harris", "Steven Sanchez", "Paul Clark",
        "Andrew Ramirez", "Joshua Lewis", "Kenneth Robinson", "Kevin Walker", "Brian Young"
    ]
    # Expand customer names list to num_customers
    while len(customer_names) < num_customers:
        first = random.choice(customer_names).split()[0]
        last = random.choice(["Hall", "Allen", "Young", "King", "Wright", "Lopez", "Hill", "Scott", "Green", "Adams", "Baker"])
        name = f"{first} {last}"
        if name not in customer_names:
            customer_names.append(name)
            
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    start_date = end_date - timedelta(days=days)
    
    # Create customer data frame
    customers_data = []
    for i in range(1, num_customers + 1):
        name = customer_names[i-1]
        email = f"{name.lower().replace(' ', '.')}@example.com"
        # Join date is spread across the 180 days
        joined_days_offset = random.randint(0, days - 10)
        joined_date = (start_date + timedelta(days=joined_days_offset)).strftime("%Y-%m-%d")
        customers_data.append({
            "customer_id": i,
            "customer_name": name,
            "email": email,
            "segment": "Unsegmented",
            "joined_date": joined_date
        })
    df_customers = pd.DataFrame(customers_data)
    
    # 2. Generate Products
    categories = {
        "Electronics": [("Laptop", 1200.0), ("Smartphone", 800.0), ("Headphones", 150.0), ("Smartwatch", 250.0), ("Monitor", 300.0)],
        "Apparel": [("T-Shirt", 25.0), ("Jeans", 60.0), ("Jacket", 120.0), ("Sneakers", 90.0), ("Socks", 10.0)],
        "Home & Kitchen": [("Coffee Maker", 80.0), ("Blender", 50.0), ("Air Fryer", 110.0), ("Vacuum Cleaner", 200.0), ("Toaster", 30.0)],
        "Books": [("Fiction Novel", 15.0), ("Science Textbook", 95.0), ("History Biography", 22.0), ("Cookbook", 30.0), ("Self-Help Book", 18.0)]
    }
    
    products_data = []
    prod_id = 1
    for cat, items in categories.items():
        for item_name, price in items:
            products_data.append({
                "product_id": prod_id,
                "product_name": f"{cat} {item_name}",
                "category": cat,
                "price": price
            })
            prod_id += 1
    df_products = pd.DataFrame(products_data)
    
    # Introduce a couple of missing prices in raw product data to simulate ETL resolving
    # We will resolve these in the pipeline
    raw_products = df_products.copy()
    raw_products.loc[raw_products['product_id'] == 3, 'price'] = np.nan # Electronics Headphones
    raw_products.loc[raw_products['product_id'] == 12, 'price'] = np.nan # Home Blender
    
    # 3. Generate Sales Transactions with Pareto Distribution
    # 20% of customers generate 80% of sales
    num_top = int(0.2 * num_customers)
    top_customer_ids = df_customers['customer_id'].values[:num_top]
    other_customer_ids = df_customers['customer_id'].values[num_top:]
    
    # Setup weights
    customer_probs = np.zeros(num_customers)
    customer_probs[:num_top] = 0.8 / num_top
    customer_probs[num_top:] = 0.2 / len(other_customer_ids)
    
    # Simulate daily transactions
    sales_data = []
    sale_id = 1
    
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        # Number of transactions per day
        num_tx = random.randint(10, 45)
        
        for _ in range(num_tx):
            # Select customer based on Pareto weights
            cust_id = np.random.choice(df_customers['customer_id'].values, p=customer_probs)
            
            # Ensure customer joined before or on purchase date
            cust_join_date = datetime.strptime(df_customers.loc[df_customers['customer_id'] == cust_id, 'joined_date'].values[0], "%Y-%m-%d")
            if current_date < cust_join_date:
                # If they hadn't joined yet, set purchase date to their join date or later
                tx_date = cust_join_date
            else:
                tx_date = current_date
                
            prod_idx = random.randint(0, len(df_products) - 1)
            prod = df_products.iloc[prod_idx]
            
            # Quantity sold (skewed towards 1 or 2)
            units = int(np.random.choice([1, 2, 3, 4, 5], p=[0.6, 0.25, 0.1, 0.04, 0.01]))
            
            sales_data.append({
                "sale_id": sale_id,
                "customer_id": int(cust_id),
                "product_id": int(prod['product_id']),
                "units_sold": units,
                # Total price will be re-calculated during ETL based on actual product price
                "total_price": None, 
                "sale_date": tx_date.strftime("%Y-%m-%d")
            })
            sale_id += 1
            
    df_sales = pd.DataFrame(sales_data)
    
    # Randomly inject some NaN units_sold to simulate cleaning
    # We will resolve this during ETL
    df_sales.loc[df_sales['sale_id'] % 250 == 0, 'units_sold'] = np.nan
    
    return df_customers, raw_products, df_sales, df_products


def run_etl_pipeline():
    """
    Run the ETL pipeline: read raw simulated data, clean, validate, and write to SQLite star schema
    """
    print("Starting ETL Pipeline...")
    
    # Generate data
    df_customers, raw_products, df_sales, df_products_clean_ref = generate_synthetic_raw_data()
    
    # 1. Clean Products
    print("Cleaning products data...")
    # Fill missing prices with the average price of their category
    for cat in raw_products['category'].unique():
        avg_price = raw_products[raw_products['category'] == cat]['price'].mean(skipna=True)
        raw_products.loc[(raw_products['category'] == cat) & (raw_products['price'].isna()), 'price'] = avg_price
    
    df_products_clean = raw_products.copy()
    
    # 2. Clean Sales
    print("Cleaning sales transaction records...")
    # Fill missing units_sold with median units_sold (1.0)
    median_units = df_sales['units_sold'].median()
    df_sales['units_sold'] = df_sales['units_sold'].fillna(median_units).astype(int)
    
    # Merge sales with products to compute total_price correctly
    df_sales_merged = df_sales.merge(df_products_clean, on='product_id', how='left')
    df_sales['total_price'] = df_sales_merged['units_sold'] * df_sales_merged['price']
    
    # Normalize dates to standard string ISO format
    df_sales['sale_date'] = pd.to_datetime(df_sales['sale_date']).dt.strftime('%Y-%m-%d')
    df_customers['joined_date'] = pd.to_datetime(df_customers['joined_date']).dt.strftime('%Y-%m-%d')
    
    # Verify no nulls remain in critical columns
    assert df_customers.isnull().sum().sum() == 0, "Nulls remain in customers table!"
    assert df_products_clean.isnull().sum().sum() == 0, "Nulls remain in products table!"
    assert df_sales.isnull().sum().sum() == 0, "Nulls remain in sales table!"
    
    print(f"Data ingestion complete. Cleaned:")
    print(f" - {len(df_customers)} customers")
    print(f" - {len(df_products_clean)} products")
    print(f" - {len(df_sales)} sales transactions")
    
    # 3. Database Ingestion
    db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "sales.db")
    
    print(f"Connecting to SQLite database: {db_path}")
    
    # Read schema.sql to initialize database
    schema_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "schema.sql")
    with open(schema_path, "r") as f:
        schema_sql = f.read()
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(schema_sql)
    conn.commit()
    conn.close()
    
    # Write tables to SQLite using SQLAlchemy engine (to avoid SQL syntax pain)
    engine = create_engine(f"sqlite:///{db_path}")
    
    # Write with if_exists='append' or 'replace'. Since we initialized with schema.sql, 
    # writing using pandas `to_sql` can be done by mapping to the created tables.
    # Note: dim_customers contains auto-increment PK, to_sql handles this if we don't write index.
    df_customers.to_sql("dim_customers", con=engine, if_exists="append", index=False)
    df_products_clean.to_sql("dim_products", con=engine, if_exists="append", index=False)
    df_sales.to_sql("fact_sales", con=engine, if_exists="append", index=False)
    
    print("Database populate successful!")
    
if __name__ == "__main__":
    run_etl_pipeline()
