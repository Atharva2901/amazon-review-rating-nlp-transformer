# AI-Powered Amazon Review Rating Predictor
### NLP • Transformers • Data Engineering • PostgreSQL • Python

This project builds an end-to-end **AI pipeline** that predicts **Amazon review star ratings (1–5)** using a fine-tuned **RoBERTa transformer model**.  
It combines **data engineering, machine learning, and NLP** to turn raw text reviews into **actionable customer insights**.

---

## What This Project Does

### 1. **Data Engineering (PostgreSQL)**
- Designed a relational database for products, customers, and reviews  
- Imported and cleaned large-scale Amazon review datasets  
- Joined tables and prepared a unified dataset for modeling  
- Exported a clean ML-ready file (`final_dataset.csv`)

---

### 2. **Exploratory Data Analysis (Python)**
- Checked rating distribution, review length patterns, and data quality  
- Identified heavy class imbalance (majority ★5 reviews)  
- Evaluated relationships between text structure and rating outcomes  

---

### 3. **Feature Engineering**
- Prepared text data for NLP modeling  
- Applied oversampling to balance classes (critical for model performance)  
- Added optional sentiment signals and category/brand features  

---

### 4. **NLP Modeling (RoBERTa Transformer)**
- Fine-tuned `roberta-base` on oversampled data  
- Used Google Colab GPU for training  
- Applied early stopping, learning rate scheduling, and evaluation strategy  
- Exported predictions + confusion matrix + performance metrics  

---

### 5. **Analysis & Insights**
Generated insights from the AI model, such as:
- Prediction accuracy and macro F1 score  
- Most common misclassifications  
- Differences between true vs predicted rating distributions  
- Which brands/categories tend to receive higher predicted satisfaction  
- Insights saved as reproducible scripts + PNG visualizations  

---

### 6. **Reporting**
The project includes:
- Confusion matrix (model strengths/weaknesses)  
- Rating distribution analysis  
- Performance over epochs  
- Business-oriented insights (brand, category, sentiment trends)

All visuals are available inside the reports/ folder.

---

## Tech Stack

**Languages:** Python, SQL  
**Database:** PostgreSQL  
**ML/NLP:** RoBERTa, HuggingFace Transformers, PyTorch, scikit-learn  
**Data Engineering:** SQLAlchemy, Pandas  
**Tools:** VS Code, Google Colab, Jupyter  
**Visualization:** seaborn, matplotlib  


---

## Key Results

- **Balanced RoBERTa model accuracy:** ~96%  
- **Macro F1 score:** ~0.95  
- Significant performance improvement after oversampling  
- Strong alignment between predicted and true rating distributions  
- Model successfully captures sentiment patterns in text reviews  

---

## Project Purpose

This project demonstrates an end-to-end real-world workflow for a **Data Analyst / ML Engineer**:

- SQL-based data modeling  
- Python EDA + feature engineering  
- NLP model training on GPUs  
- Insights generation for business decision-making  
- Clean, reproducible, production-ready structure  

---

## Author

**Atharva Deshmukh**  
M.S. Computer Science, Stevens Institute of Technology  
GitHub: https://github.com/Atharva2901  
LinkedIn: https://www.linkedin.com/m/in/atharva-deshmukh-0968751a6/


