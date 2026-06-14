# DecodeLab Data Science Internship - Week 2
## Credit Card Fraud Detection using Machine Learning

---

## Project Overview

This repository contains my Week 2 internship submission for the DecodeLab Data Science Internship program. The project focuses on building a machine learning model to detect fraudulent credit card transactions using a real-world imbalanced dataset.

---

## Dataset

**File:** `creditcard.csv`

The dataset contains anonymized credit card transaction records, including:

- V1 to V28 — Anonymized PCA features
- Time — Seconds elapsed between transactions
- Amount — Transaction value
- Class — Target variable (0 = Legitimate, 1 = Fraud)

---

## Tasks Completed

**Task 1: Data Loading & Understanding**

- Loaded the dataset using Pandas
- Explored dataset dimensions and structure
- Identified columns, data types, and class distribution
- Observed severe class imbalance between fraud and legitimate transactions

**Task 2: Data Preprocessing**

- Checked for and confirmed no missing values
- Scaled `Time` and `Amount` features using StandardScaler
- Separated features and target variable into X and y

**Task 3: Handling Class Imbalance using SMOTE**

- Applied SMOTE to generate synthetic samples for the minority (fraud) class
- Balanced the dataset before model training

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

**Task 4: Model Building**

- Split the balanced dataset into training and testing sets
- Trained the following models:
  - Logistic Regression
  - Random Forest Classifier

**Task 5: Model Evaluation**

- Evaluated models using:
  - Accuracy
  - Precision
  - Recall (critical for fraud detection)
  - F1-Score
  - Confusion Matrix

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib / Seaborn
- Google Colab
- GitHub

---

## Repository Structure
