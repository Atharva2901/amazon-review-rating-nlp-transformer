# scripts/insights_generator.py
import pandas as pd

df = pd.read_csv("data/model_predictions.csv")

acc = (df["true_rating"] == df["predicted_rating"]).mean() * 100
print(f"✅ Model Accuracy: {acc:.2f}%")

# common errors
mis = df[df["true_rating"] != df["predicted_rating"]]
top_errors = (
    mis.groupby(["true_rating", "predicted_rating"])
      .size()
      .reset_index(name="count")
      .sort_values("count", ascending=False)
      .head(5)
)
print("\n🔁 Top misclassifications:")
print(top_errors)
