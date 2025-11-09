from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# Encode special characters like @
DB_PASS = DB_PASS.replace("@", "%40")

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

def test_connection():
    try:
        with engine.connect() as conn:
            print("✅ Connected to PostgreSQL successfully!")
    except Exception as e:
        print("❌ Connection failed:", e)

if __name__ == "__main__":
    test_connection()

query = """
SELECT r.review_id, r.rating, r.review_text, r.review_title, r.review_date,
       p.brand, p.category, c.username
FROM reviews r
JOIN products p ON r.product_id = p.product_id
JOIN customers c ON r.customer_id = c.customer_id;
"""

df = pd.read_sql(query, engine)
print("✅ Data loaded:", len(df), "rows")
df.head()

# df.to_csv("data/raw_reviews.csv", index=False)

import seaborn as sns
import matplotlib.pyplot as plt

print(df.info())
print(df.describe())
print(df['rating'].value_counts())

sns.countplot(x='rating', data=df)
plt.title("Rating Distribution")
plt.show()

import re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['review_text'].apply(clean_text)
# df.to_csv("data/clean_reviews.csv", index=False)
print("✅ Cleaned dataset saved!")

brand_features = pd.read_csv("data/brand_insights.csv")
user_features = pd.read_csv("data/customer_insights.csv")

brand_features.to_sql('brand_insights', engine, if_exists='replace', index=False)
user_features.to_sql('customer_insights', engine, if_exists='replace', index=False)
print("✅ Feature tables stored in PostgreSQL")


