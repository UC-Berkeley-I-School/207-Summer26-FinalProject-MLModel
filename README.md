# ✈️ SFO Flight Delay Prediction with XGBoost

## Overview

Flight delays can create significant disruptions for passengers, airlines, and airport operations. This project explores whether machine learning can be used to predict flight delays at **San Francisco International Airport (SFO)** using historical flight and related operational data.

We developed and evaluated an **XGBoost classification model** to identify flights that are at risk of being delayed. The project involved the full machine learning workflow, including data extraction, preprocessing, exploratory data analysis, feature engineering, baseline modeling, model improvement, and evaluation.

This project was completed collaboratively as part of **UC Berkeley's Master of Information and Data Science (MIDS) program**.

---

## 🎯 Project Objective

The goal of this project was to answer:

> **Can we predict whether a flight departing from SFO will be delayed using information available prior to departure?**

A successful model could help travelers, airlines, and airport operations better anticipate disruptions and make more informed decisions.

---

## 🧠 Machine Learning Approach

We approached the problem as a **binary classification task**, where flights were classified as either:

- **Delayed**
- **Not Delayed**

We first established a baseline model and then developed an improved model using **XGBoost**, an ensemble tree-based machine learning algorithm well suited for structured/tabular data.

Our workflow included:

1. **Data Extraction**
2. **Data Cleaning & Preprocessing**
3. **Exploratory Data Analysis (EDA)**
4. **Feature Engineering**
5. **Baseline Model Development**
6. **XGBoost Model Development**
7. **Hyperparameter Tuning**
8. **Model Evaluation**
9. **Final Model Selection**

---

## 📊 Model Evaluation

Model performance was evaluated using classification metrics such as:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Particular attention was given to the model's ability to correctly identify delayed flights rather than relying solely on overall accuracy.

## 🛠️ Technologies

**Languages & Tools**

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn
- Git / GitHub

---

## 📁 Repository Structure

```text
├── Baseline_Model/
├── Data_Extraction/
├── Data_Preprocessing/
├── EDA/
├── Improved_Model/
├── Final_Version/
└── README.md
