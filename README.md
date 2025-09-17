# 🧬 Breast Cancer Classification Models

## 📌 Overview
This project applies different **machine learning models** to the **Breast Cancer dataset**, predicting whether a tumor is **benign** or **malignant**.  

Each model is implemented in a **separate file** so you can compare their performance individually.

---

## 📂 Files

### 1️⃣ `logisticRegModel.py`
- Uses **Logistic Regression**  
- Steps:
  - Load and preprocess dataset
  - Train Logistic Regression model
  - Evaluate with:
    - Confusion Matrix

---

### 2️⃣ `XGBoostModel.py`
- Uses **XGBoost Classifier**  
- Steps:
  - Requires minimal preprocessing
  - Train using gradient boosting
  - Evaluate with:
    - Confusion Matrix
    - Accuracy & CV Scores

---

### 3️⃣ `CatBoostModel.py`
- Uses **CatBoost Classifier**  
- Steps:
  - Gradient boosting optimized for categorical data
  - Evaluate with:
    - Confusion Matrix
    - Accuracy & CV Scores

---

## 📂 Dataset
The `breast_cancer.csv` file contains:
- **Tumor characteristics (features)**  
- **Diagnosis class** (target variable: class)

---

## 🛠 Requirements
- Python 3.x  
- pandas  
- scikit-learn  
- xgboost  
- catboost  
