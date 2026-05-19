# 🩺 GlycoAID – Diabetes Risk Prediction System
GlycoAID is a machine learning-powered web application designed to assist healthcare professionals in early detection of diabetes risk using routine medical data.
This project was developed as part of an AI in Healthcare assignment to demonstrate the application of supervised learning in clinical decision support.

## 📌 Project Objective

To build and deploy a machine learning system that predicts whether a patient is at risk of diabetes based on diagnostic health features.

## 📊 Dataset

- Source:  
  https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv

- The dataset contains medical records with the following features:

| Feature | Description |
|--------|------------|
| Pregnancies | Number of pregnancies |
| Glucose | Blood glucose level |
| BloodPressure | Blood pressure value |
| SkinThickness | Skin fold thickness |
| Insulin | Insulin level |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Genetic diabetes likelihood |
| Age | Patient age |

**Target Variable:**
- `0` → Low Risk (No Diabetes)  
- `1` → High Risk (Diabetes)

## ⚙️ Machine Learning Approach

- Problem Type: **Supervised Learning (Binary Classification)**

### Models Used:
- Logistic Regression  
- Decision Tree Classifier  

### Workflow:
- Data loading from URL  
- Data cleaning (handling invalid zero values)  
- Exploratory Data Analysis (EDA)  
- Feature scaling  
- Train-test split  
- Model training and evaluation  
- Model comparison  
- Model saving using Joblib 
## 📈 Model Performance

| Model | Accuracy |
|------|--------|
| Logistic Regression | ~0.78 |
| Decision Tree | ~0.72 |

Logistic Regression was selected as the best-performing model.

## 🚀 Streamlit Web Application

The application allows medical staff to:

- Input patient medical details  
- Get real-time diabetes risk prediction  
- View probability score  
- Understand feature influence (explainability)  
