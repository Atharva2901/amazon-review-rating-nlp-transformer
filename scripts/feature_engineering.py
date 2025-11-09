import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://postgres:Atharva%40123@localhost:5432/amazon_reviews_ai")

query = """
SELECT r.rating, r.review_text, r.review_date, 
       p.brand, p.category, u.avg_sentiment
FROM reviews r
JOIN products p ON r.product_id = p.product_id
JOIN customers c ON r.customer_id = c.customer_id
JOIN customer_insights u ON c.username = u.username;
"""

df = pd.read_sql(query, engine)
print(df.head())
print(df.shape)


# Save a CSV bridge for Colab
df.to_csv("data/final_dataset.csv", index=False)
print("✅ Final dataset exported for model training.")
