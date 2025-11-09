import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://postgres:Atharva%40123@localhost:5432/amazon_reviews_ai")

preds = pd.read_csv("data/model_predictions.csv")
preds.to_sql("nlp_predictions", engine, if_exists="replace", index=False)
print("✅ Predictions stored in PostgreSQL")
