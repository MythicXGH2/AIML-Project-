# 🎓 Hostel Expense Predictor (ML-Based)

A Machine Learning application designed to help hostel students estimate their monthly expenses based on lifestyle habits, academic year, and social activities.

## 📌 Project Overview
Managing finances in a hostel is a challenge for many students. This project uses a **Random Forest Regressor** to predict monthly spending by analyzing various features like canteen visits, online shopping, and even part-time job status. It includes a full pipeline: from synthetic data generation to hyperparameter tuning.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** * `Pandas` & `NumPy` (Data Manipulation)
    * `Scikit-Learn` (Machine Learning Pipeline)
    * `Matplotlib` & `Seaborn` (Data Visualization)
    * `Joblib` (Model Persistence)

## 🧮 How It Works
The model is trained on a synthetic dataset designed to mimic realistic student spending patterns. It utilizes a **Linear-Non-Linear Hybrid Logic**:
* **Base Costs:** Fixed costs like mobile recharge and subscriptions.
* **Variable Costs:** Weekly canteen visits, transport, and stationery.
* **Non-Linear Interactions:** The model accounts for "Seniority bias" (Year 3 & 4 students spending more on outside food) and "Social Index" (interaction between outings and meals).

### The Math Behind the Prediction
The target variable $Monthly\_Expense$ is calculated as:
$$Expense = \sum (Feature_{i} \times Weight_{i}) + \text{Social\_Index} + \epsilon$$
*Where $\epsilon$ represents random noise (unforeseen expenses).*

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python installed. Then, install the required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
