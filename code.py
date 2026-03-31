import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# Data Generation
def generate_student_data(n=1000):
    np.random.seed(42)
    data = {
        'canteen_visits': np.random.randint(0, 21, n),
        'outside_meals': np.random.randint(0, 10, n),
        'online_orders': np.random.randint(0, 8, n),
        'transport_trips': np.random.randint(0, 15, n),
        'recharge_type': np.random.choice(['Budget', 'Standard', 'Premium'], n),
        'subscriptions': np.random.randint(0, 4, n),
        'stationery': np.random.randint(50, 600, n),
        'weekend_outings': np.random.randint(0, 5, n),
        'has_part_time': np.random.choice([0, 1], n, p=[0.8, 0.2]),
        'year_of_study': np.random.randint(1, 5, n),
    }
    df = pd.DataFrame(data)

    # Mapping recharge to actual cost
    recharge_map = {'Budget': 199, 'Standard': 299, 'Premium': 499}
    
    # Advanced logic: Seniors (Year 3/4) tend to spend 15% more on outside food
    senior_multiplier = df['year_of_study'].apply(lambda x: 1.15 if x > 2 else 1.0)

    df['monthly_expense'] = (
        (df['canteen_visits'] * 4 * 75) +
        (df['outside_meals'] * 4 * 180 * senior_multiplier) +
        (df['online_orders'] * 400) +
        (df['transport_trips'] * 50) +
        (df['recharge_type'].map(recharge_map)) +
        (df['subscriptions'] * 199) +
        (df['stationery']) +
        (df['weekend_outings'] * 4 * 400) -
        (df['has_part_time'] * 2000) +
        np.random.normal(0, 400, n)
    ).clip(2000, 15000).round(0)
    
    return df

df = generate_student_data()

# Feature Engineering
df['social_index'] = df['weekend_outings'] * df['outside_meals']

# The Pipeline (Preprocessing + Model)
X = df.drop('monthly_expense', axis=1)
y = df['monthly_expense']

numeric_features = ['canteen_visits', 'outside_meals', 'online_orders', 'transport_trips', 
                    'subscriptions', 'stationery', 'weekend_outings', 'year_of_study', 'social_index']
categorical_features = ['recharge_type', 'has_part_time']

# Preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Hyperparameter Tuning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Evaluation 
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"--- Model Performance ---")
print(f"MAE:  ₹{mae:.2f}")
print(f"RMSE: ₹{rmse:.2f}")
print(f"R²:   {r2:.4f}")

# Save Model
joblib.dump(best_model, 'student_expense_model.pkl')
print("\nModel saved as 'student_expense_model.pkl'")

# Inference Function 
def predict_expense(input_dict):
    model = joblib.load('student_expense_model.pkl')
    input_df = pd.DataFrame([input_dict])
    input_df['social_index'] = input_df['weekend_outings'] * input_df['outside_meals']
    return model.predict(input_df)[0]

# Example
test_student = {
    'canteen_visits': 12,
    'outside_meals': 4,
    'online_orders': 1,
    'transport_trips': 5,
    'recharge_type': 'Standard',
    'subscriptions': 2,
    'stationery': 300,
    'weekend_outings': 1,
    'has_part_time': 0,
    'year_of_study': 2
}

print(f"Predicted Expense for Test Student: ₹{predict_expense(test_student):.2f}")
