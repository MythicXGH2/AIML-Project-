# Project Report: Hostel Expense Predictor (ML-Based)
**Course:** Bring Your Own Project (BYOP)  
**Student Name:** Patel Meghal Vipulkumar 
**Registration Number:** 25BAS10001 

---

## 1. Introduction & Problem Statement
### 1.1 The Problem
For many students, transitioning to hostel life involves managing a personal budget for the first time. Many students face a "financial crunch" during the final week of the month because they fail to account for how micro-habits (like canteen visits or online shopping) accumulate. 

### 1.2 The Solution
The **Hostel Expense Predictor** is a proactive financial tool. Unlike traditional expense trackers that look backward at what was spent, this Machine Learning model looks forward. By inputting a student's weekly habits and academic year, it predicts the total monthly expenditure, allowing for better financial planning.

---

## 2. Methodology
### 2.1 Data Generation & Features
Since real student financial data is private and difficult to aggregate, I developed a synthetic dataset of 1,000 students. The data mimics real-world distributions observed on campus.

**Key Features:**
- **Canteen Visits & Outside Meals:** Captured as weekly frequencies.
- **Academic Seniority (Year of Study):** Accounts for shifts in spending as students progress (e.g., more coffee/internship-related travel).
- **Social Index:** An engineered feature combining weekend outings with dining habits to capture "social spending."
- **Part-Time Status:** A binary feature that acts as a negative weight (income).

### 2.2 The Machine Learning Pipeline
The project utilizes a **Random Forest Regressor**. This model was chosen over Linear Regression because spending habits are rarely linear. For instance, the jump in cost from "2 outside meals" to "4 outside meals" often includes higher transport and social costs that a simple line cannot capture.



**Preprocessing Steps:**
1. **Standardization:** Using `StandardScaler` to ensure all numerical features (like 15 canteen visits vs 1 part-time job) are on the same scale.
2. **One-Hot Encoding:** Converting categorical data like `recharge_type` into a format the model can understand.

---

## 3. Implementation Details
The project was built using a modular Python structure:
- **`pandas` & `numpy`**: For data structuring and math.
- **`scikit-learn`**: For the core ML pipeline, including `ColumnTransformer` for automated preprocessing.
- **`joblib`**: Used to save the trained model (`.pkl`) so it can be deployed in an app without retraining.

---

## 4. Challenges & Key Learnings
### 4.1 Technical Challenges
- **Library Mismatch:** I initially faced a `ModuleNotFoundError` for Pandas. I resolved this by correctly configuring the Python Interpreter in VS Code and using `python -m pip install` to target the active environment.
- **Feature Interaction:** I learned that some features are more powerful when combined. Creating the `social_index` significantly increased the model's accuracy.

### 4.2 Reflection
This project taught me that **data preprocessing** is just as important as the model itself. Without proper scaling and encoding, even a powerful Random Forest model would produce inaccurate results.

---

## 5. Evaluation & Results
The model was evaluated using two primary metrics:
- **Mean Absolute Error (MAE):** The model predicts within a margin of error of approximately ₹350–₹500, which is highly acceptable for a student budget.
- **R² Score:** The model achieved an R² score of ~0.92, indicating that 92% of the variance in spending is explained by the features provided.



---

## 6. Conclusion
The **Hostel Expense Predictor** successfully demonstrates how Machine Learning can solve everyday problems. By turning lifestyle habits into data points, we can provide students with the clarity they need to manage their finances effectively.
