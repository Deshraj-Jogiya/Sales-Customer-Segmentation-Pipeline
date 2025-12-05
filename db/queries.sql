-- queries.sql: Analytical SQL queries for sales and RFM customer segmentation analysis

-- 1. Category Revenue Share
-- Calculates total revenue and percentage share for each product category
SELECT 
    p.category,
    SUM(s.total_price) AS total_revenue,
    ROUND(SUM(s.total_price) * 100.0 / (SELECT SUM(total_price) FROM fact_sales), 2) AS revenue_share_percentage
FROM fact_sales s
JOIN dim_products p ON s.product_id = p.product_id
GROUP BY p.category
ORDER BY total_revenue DESC;


-- 2. Top Customer Segments by Spending
-- Analyzes spending behavior and customer count across K-Means identified segments
SELECT 
    c.segment,
    COUNT(DISTINCT c.customer_id) AS total_customers,
    ROUND(SUM(s.total_price), 2) AS total_spending,
    ROUND(AVG(s.total_price), 2) AS avg_sale_value,
    ROUND(SUM(s.total_price) / COUNT(DISTINCT c.customer_id), 2) AS average_spending_per_customer
FROM dim_customers c
LEFT JOIN fact_sales s ON c.customer_id = s.customer_id
GROUP BY c.segment
ORDER BY total_spending DESC;


-- 3. Monthly Active Customers and Revenue
-- Tracks unique customers purchasing and total revenue generated month-over-month
SELECT 
    strftime('%Y-%m', s.sale_date) AS sales_month,
    COUNT(DISTINCT s.customer_id) AS active_customers,
    ROUND(SUM(s.total_price), 2) AS monthly_revenue,
    ROUND(SUM(s.units_sold), 2) AS monthly_units_sold
FROM fact_sales s
GROUP BY sales_month
ORDER BY sales_month ASC;


-- 4. Top 10 High-Value Customers (Monetary Leaderboard)
-- Retrieves the highest-spending customers, showing their details, segments, and purchase count
SELECT 
    c.customer_id,
    c.customer_name,
    c.email,
    c.segment,
    COUNT(s.sale_id) AS total_orders,
    SUM(s.units_sold) AS total_units_purchased,
    ROUND(SUM(s.total_price), 2) AS total_spent
FROM dim_customers c
JOIN fact_sales s ON c.customer_id = s.customer_id
GROUP BY c.customer_id, c.customer_name, c.email, c.segment
ORDER BY total_spent DESC
LIMIT 10;


-- 5. Monthly Customer Join cohort sizing
-- Size of new customer cohort grouped by their join date month and current segments
SELECT 
    strftime('%Y-%m', joined_date) AS join_month,
    segment,
    COUNT(customer_id) AS cohort_size
FROM dim_customers
GROUP BY join_month, segment
ORDER BY join_month ASC, cohort_size DESC;


-- 6. Product Sales and Revenue Contribution
-- Provides a performance summary of all products sold, sorted by total units sold
SELECT 
    p.product_id,
    p.product_name,
    p.category,
    p.price AS unit_price,
    SUM(s.units_sold) AS total_units_sold,
    ROUND(SUM(s.total_price), 2) AS total_revenue
FROM dim_products p
LEFT JOIN fact_sales s ON p.product_id = s.product_id
GROUP BY p.product_id, p.product_name, p.category, p.price
ORDER BY total_units_sold DESC;
