import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.title("Amazon Review → AI Rating")

review = st.text_area("Paste a review:")
if st.button("Predict"):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=5)
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item() + 1
    st.success(f"Predicted Rating: ⭐ {pred}")