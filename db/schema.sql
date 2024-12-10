-- schema.sql: SQLite star schema database for sales analytics pipeline

-- Dimension Table: Customers
CREATE TABLE IF NOT EXISTS dim_customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    segment TEXT DEFAULT 'Unsegmented',
    joined_date DATE NOT NULL
);

-- Dimension Table: Products
CREATE TABLE IF NOT EXISTS dim_products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL
);

-- Fact Table: Sales
CREATE TABLE IF NOT EXISTS fact_sales (
    sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    units_sold INTEGER NOT NULL,
    total_price REAL NOT NULL,
    sale_date DATE NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES dim_customers (customer_id),
    FOREIGN KEY (product_id) REFERENCES dim_products (product_id)
);
