# 🩺 Stroke Prediction Model using R

## 🚀 Overview
This project implements **machine learning models in R** to predict the risk of stroke in patients based on demographic and medical history data. The model helps in **early detection** and **preventive care**.

## 📊 Dataset
The dataset contains **11 clinical features** (age, hypertension, heart disease, glucose level, smoking status, etc.) and a binary outcome (`stroke: 0/1`).

## 🔬 Models Implemented
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **XGBoost** (Best performing model)

## 📈 Performance Metrics
| **Model**               | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|-------------------------|-------------|--------------|------------|--------------|
| **Logistic Regression** | **94.6%**    | **94.6%**    | **100%**   | **97.2%**    |
| **Decision Tree**       | **94.5%**    | **94.5%**    | **100%**   | **97.1%**    |
| **Random Forest**       | **94.5%**    | **94.5%**    | **100%**   | **97.1%**    |

## 🔍 Key Insights
- **Age, Hypertension, and Heart Disease** are the strongest predictors of stroke.
- **XGBoost performed best** but can be further improved with **hyperparameter tuning**.
- **Class imbalance** in the dataset was handled, but further improvements using **SMOTE** can enhance results.

## 💻 Installation & Usage
### **1️⃣ Clone the repository**
```bash
git clone https://github.com/Sadanandgoud/stroke-prediction-r.git
cd stroke-prediction-r
install.packages(c("tidyverse", "caret", "e1071", "randomForest", "xgboost", "pROC", "rmarkdown"))
source("stroke_prediction_model.R")


