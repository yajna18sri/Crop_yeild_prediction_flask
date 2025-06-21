import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle

# 🔁 Replace this with your actual data file
data = pd.read_csv("crop_yield.csv")  # Make sure the file is in the same folder

# ✅ Define features and target
X = data[['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
y = data['Yield']

# ✅ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define preprocessing
categorical_cols = ['Crop', 'Season', 'State']
numerical_cols = ['Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

# ✅ Create and train pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', DecisionTreeRegressor(random_state=42))
])

pipeline.fit(X_train, y_train)

# ✅ Save model and preprocessor separately
with open("dtr.pkl", "wb") as f:
    pickle.dump(pipeline.named_steps['model'], f)

with open("preprocessor.pkl", "wb") as f:
    pickle.dump(pipeline.named_steps['preprocess'], f)

print("✅ Model and preprocessor saved using scikit-learn", pipeline.named_steps['model'].__module__)
