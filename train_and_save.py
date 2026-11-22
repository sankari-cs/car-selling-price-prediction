# train_and_save.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Load & clean
df = pd.read_csv("car data.csv")
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
df = df.drop_duplicates().reset_index(drop=True)
df = df.dropna(subset=["selling_price"]).reset_index(drop=True)

# 2. Feature engineering
CURRENT_YEAR = 2024
df["car_age"] = CURRENT_YEAR - df["year"]
df = df.drop(columns=["year", "car_name"])

# 3. Encode categoricals
cat_cols = df.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"Encoder for '{col}': {list(le.classes_)}")

# 4. Train
X = df.drop(columns=["selling_price"])
y = df["selling_price"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
}
best_name, best_model, best_r2 = None, None, -999
print("\nModel comparison:")
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    r2 = r2_score(y_test, p)
    mae = mean_absolute_error(y_test, p)
    rmse = np.sqrt(mean_squared_error(y_test, p))
    print(f"  {name:20s} MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
    if r2 > best_r2:
        best_r2, best_name, best_model = r2, name, m

print(f"\nBest model: {best_name} (R2={best_r2:.4f})")

# 5. Save
with open("model.pkl", "wb") as f: pickle.dump(best_model, f)
with open("encoders.pkl", "wb") as f: pickle.dump(encoders, f)
with open("feature_names.pkl", "wb") as f: pickle.dump(feature_names, f)
print("\n✅ Saved: model.pkl, encoders.pkl, feature_names.pkl")