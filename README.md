# Hostel Student Monthly Expense Predictor

A machine learning project that predicts the monthly expenditure of hostel
students based on their daily lifestyle habits.

## Problem
Hostel students often overspend without realising it. This model helps predict
monthly expenses from habits like canteen visits, online shopping, and outings —
enabling better financial planning.

## Dataset
Synthetically generated data (500 students) based on realistic hostel spending
patterns observed at VIT Bhopal. Features mirror common student expenditure
categories.

## Features Used
- Canteen visits per week
- Outside food orders per week
- Online shopping orders per month
- Transport trips
- Mobile recharge amount
- Subscription services (OTT, etc.)
- Stationery spend
- Weekend outings
- Part-time job (yes/no)
- Year of study

## Models
| Model               | MAE     | R²    |
|---------------------|---------|-------|
| Linear Regression   | ~₹280   | ~0.91 |
| Random Forest       | ~₹180   | ~0.96 |

## How to Run
1. Clone this repo
2. `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Open `expense_predictor.ipynb` in Jupyter
4. Run all cells
