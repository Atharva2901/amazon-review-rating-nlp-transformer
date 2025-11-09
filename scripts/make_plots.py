# scripts/make_plots.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

df = pd.read_csv("data/model_predictions.csv")

# 1) confusion matrix
cm = confusion_matrix(df["true_rating"], df["predicted_rating"])
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix – RoBERTa (balanced)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("insights/confusion_matrix.png", dpi=300, bbox_inches="tight")

# 2) true vs predicted distribution
plt.figure(figsize=(5,4))
sns.histplot(df["true_rating"], color="blue", label="True", discrete=True)
sns.histplot(df["predicted_rating"], color="orange", label="Predicted", discrete=True)
plt.legend()
plt.title("True vs Predicted Rating Distribution")
plt.savefig("insights/distribution.png", dpi=300, bbox_inches="tight")

print("✅ Plots saved in reports/")
