# 🤖 DecodeLab Data Science Internship – Week 2
## Credit Card Fraud Detection using Machine Learning

---

## 📌 Project Overview

This project is part of the **DecodeLab Data Science Internship – Week 2**.

The objective is to build a machine learning model to detect fraudulent credit card transactions using the dataset `creditcard.csv`.

Since the dataset is highly imbalanced, **SMOTE** is used to balance the classes before training the model.

---

## 📂 Dataset

**File:** `creditcard.csv`

| Feature | Description |
|--------|-------------|
| V1 – V28 | Anonymized PCA features |
| Time | Transaction time |
| Amount | Transaction value |
| Class | Target variable (0 = Legit, 1 = Fraud) |

---

## ⚠️ Problem Statement

This is a highly imbalanced classification problem where fraud cases are very rare.  
The goal is to correctly detect fraudulent transactions with high recall.

---

## 🧹 Data Preprocessing

- Loaded dataset using Pandas
- Checked dataset structure and shape
- No missing values found
- Scaled `Time` and `Amount` using `StandardScaler`
- Split features and target into `X` and `y`

---

## ⚖️ SMOTE – Handling Class Imbalance

SMOTE was applied to balance the dataset by generating synthetic samples of the minority (fraud) class.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

---

## ✂️ Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)
```

---

## 🤖 Model Building

Models trained on the SMOTE-balanced dataset:

- ✅ Logistic Regression
- ✅ Random Forest Classifier


## 📊 Evaluation Metrics

| Metric | Why It Matters |
|--------|----------------|
| Accuracy | Overall correctness |
| Precision | How many flagged frauds are real |
| **Recall** | **Critical — catches actual fraud cases** |
| F1-Score | Balance of precision and recall |
| Confusion Matrix | Visual breakdown of predictions |


## 🛠️ Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib / Seaborn
- Google Colab
- GitHub

---

## 📁 Repository Structure
