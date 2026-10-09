import pandas as pd
import numpy as np
import joblib
import os

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv("train.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

# Time features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
df["Day"] = df["Date"].dt.day
df["Quarter"] = df["Date"].dt.quarter

df["IsHoliday"] = df["IsHoliday"].astype(int)

# Lag features
df["Lag_1"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)
df["Lag_2"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(2)
df["Lag_4"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(4)

# Rolling mean
df["Rolling_Mean_4"] = (
    df.groupby(["Store", "Dept"])["Weekly_Sales"]
    .transform(lambda x: x.shift(1).rolling(4).mean())
)

# Remove missing rows
df = df.dropna().reset_index(drop=True)

# Features
features = [
    "Store",
    "Dept",
    "Year",
    "Month",
    "Week",
    "Day",
    "Quarter",
    "IsHoliday",
    "Lag_1",
    "Lag_2",
    "Lag_4",
    "Rolling_Mean_4"
]

X = df[features]
y = df["Weekly_Sales"]

# Chronological split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# XGBoost
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

print("Training XGBoost model...")

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\nModel Training Completed!")
print("-----------------------------")
print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/xgboost_model.pkl")

# Save feature list
joblib.dump(features, "models/xgboost_features.pkl")

print("\nModel saved successfully!")
print("models/xgboost_model.pkl")