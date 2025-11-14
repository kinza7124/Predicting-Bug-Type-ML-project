# Bug Type Classification Using NLP & Machine Learning
### Automatically Predicting Software Bug Types from Text + Metadata

---

## 📌 Overview
This project builds a **Machine Learning + NLP pipeline** to classify software bug reports into different **Bug Types**.  
It includes **EDA**, preprocessing, NLP feature extraction, model training and visualizations.

---

## 🎯 Project Objectives
- Perform EDA & descriptive analysis  
- Clean, preprocess, and encode features  
- Apply NLP techniques to textual columns  
- Train multiple ML models  
- Evaluate performance using different metrics  
- Generate a final unseen testing file  
- Build a complete ML workflow ready for deployment  

---

## Data Preprocessing Steps

### ✔ Cleaning & Formatting
- Removed missing values  
- Standardized text (lowercase, punctuation removal, trimming)  
- Removed duplicate rows  
- Categorical + numerical preprocessing  

### ✔ NLP Techniques Used
- One Hot Encoder  
- TF-IDF Vectorization  
- CountVectorizer

### ✔ Feature Encoding
- Label Encoding for target (Bug Type)  
- Label Encoding for product/component categories  
- StandardScaler for numeric columns  

---

## 📊 Exploratory Data Analysis (EDA)
Included analysis & visualizations for:
- Bug type frequency  
- Year- and month-wise trends  
- Product group distributions  
- Component distributions  
- Correlation matrix  
- Heatmaps  
- Histograms / bar graphs  

Visualization tools used:
```python
import matplotlib.pyplot as plt
import seaborn as sns
```
---

# Machine Learning Models Used

🔹 Base Models

- Logistic Regression

- KNN

- Decision Tree

- Random Forest

- SVM

- Gaussian/MultiNomial Naive Bayes

- Gradient Boosting Classifier

🔹 Ensemble Models

Voting Classifier

Stacking Classifier

🔹 KNN Distance Metrics Used

- Cosine

# 📈 Model Evaluation

Metrics used:

- Accuracy

- Precision

- Recall

- F1-Score

- Confusion Matrix

- Classification Report
