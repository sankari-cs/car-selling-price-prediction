# =============================================================
# CAR SELLING PRICE PREDICTION - Full Project
# Run: python analysis.py
# =============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
os.makedirs("plots", exist_ok=True)

# =============================================================
# SECTION 1 - LOAD DATA
# =============================================================
print("\n" + "="*60)
print("SECTION 1 - LOADING DATA")
print("="*60)

# Try common file names
candidates = ["car data.csv", "car_data.csv", "CAR DETAILS FROM CAR DEKHO.csv",
              "cardata.csv", "car.csv"]
df = None
for f in candidates:
    if os.path.exists(f):
        df = pd.read_csv(f)
        print(f"Loaded: {f}")
        break

if df is None:
    raise FileNotFoundError(
        "CSV not found in this folder. Put the car CSV here and rename to 'car data.csv'."
    )

print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head())

# =============================================================
# SECTION 2 - INSPECT
# =============================================================
print("\n" + "="*60)
print("SECTION 2 - INSPECTION")
print("="*60)
print(df.info())
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicates:", df.duplicated().sum())
print("\nDescribe:\n", df.describe(include="all"))

# =============================================================
# SECTION 3 - CLEANING
# =============================================================
print("\n" + "="*60)
print("SECTION 3 - CLEANING")
print("="*60)

# Standardize column names (some datasets use different cases/underscores)
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
print("Normalized columns:", list(df.columns))

# Drop exact duplicates
before = df.shape[0]
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dropped {before - df.shape[0]} duplicate rows")

# Drop any rows with missing target
target_candidates = ["selling_price", "price", "sellingprice"]
TARGET = next((t for t in target_candidates if t in df.columns), None)
if TARGET is None:
    raise ValueError(f"Target column not found. Columns: {list(df.columns)}")
print(f"Target column: {TARGET}")

df = df.dropna(subset=[TARGET]).reset_index(drop=True)

# =============================================================
# SECTION 4 - EDA
# =============================================================
print("\n" + "="*60)
print("SECTION 4 - EDA")
print("="*60)

# 4.1 Target distribution
plt.figure(figsize=(8,4))
sns.histplot(df[TARGET], kde=True, bins=30, color="steelblue")
plt.title(f"Distribution of {TARGET}")
plt.tight_layout(); plt.savefig("plots/01_target_dist.png", dpi=120); plt.show()

# 4.2 Correlation heatmap
num_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(9,6))
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout(); plt.savefig("plots/02_corr.png", dpi=120); plt.show()

# 4.3 Boxplots for categorical vs target
cat_cols = df.select_dtypes(include="object").columns.tolist()
if TARGET in cat_cols: cat_cols.remove(TARGET)

for i, col in enumerate(cat_cols):
    if df[col].nunique() <= 15:
        plt.figure(figsize=(7,4))
        sns.boxplot(x=col, y=TARGET, data=df)
        plt.title(f"{TARGET} by {col}")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f"plots/03_box_{i}_{col}.png", dpi=120)
        plt.show()

# 4.4 Scatter: Present price vs Selling price (if exists)
if "present_price" in df.columns:
    plt.figure(figsize=(7,4))
    sns.scatterplot(x="present_price", y=TARGET, data=df, alpha=0.6)
    plt.title("Present Price vs Selling Price")
    plt.tight_layout(); plt.savefig("plots/04_scatter.png", dpi=120); plt.show()

# =============================================================
# SECTION 5 - FEATURE ENGINEERING
# =============================================================
print("\n" + "="*60)
print("SECTION 5 - FEATURE ENGINEERING")
print("="*60)

df_fe = df.copy()

# 5.1 Age of car from Year
if "year" in df_fe.columns:
    CURRENT_YEAR = 2024
    df_fe["car_age"] = CURRENT_YEAR - df_fe["year"]
    df_fe = df_fe.drop(columns=["year"])
    print("Created 'car_age' from 'year'")

# 5.2 Drop Car_Name (too many unique values; adds noise without strong signal)
for c in ["car_name", "name"]:
    if c in df_fe.columns:
        df_fe = df_fe.drop(columns=[c])
        print(f"Dropped '{c}'")

# 5.3 Encode categorical columns
le = LabelEncoder()
encoded_cols = []
for col in df_fe.select_dtypes(include="object").columns:
    df_fe[col] = le.fit_transform(df_fe[col])
    encoded_cols.append(col)
    print(f"Label-encoded: {col}")

print("\nFinal feature columns:", list(df_fe.columns))

# =============================================================
# SECTION 6 - TRAIN/TEST SPLIT + MODELS
# =============================================================
print("\n" + "="*60)
print("SECTION 6 - TRAINING MODELS")
print("="*60)

X = df_fe.drop(columns=[TARGET])
y = df_fe[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
}

results = []
trained = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    trained[name] = model
    print(f"{name:22s} | MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\nModel Comparison:\n", results_df.to_string(index=False))

# =============================================================
# SECTION 7 - ACTUAL VS PREDICTED (best model)
# =============================================================
print("\n" + "="*60)
print("SECTION 7 - VISUALIZING BEST MODEL")
print("="*60)

best_name = results_df.iloc[0]["Model"]
best_model = trained[best_name]
print(f"Best model: {best_name}")

preds = best_model.predict(X_test)
plt.figure(figsize=(7,5))
plt.scatter(y_test, preds, alpha=0.6, color="teal")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual"); plt.ylabel("Predicted")
plt.title(f"Actual vs Predicted ({best_name})")
plt.tight_layout(); plt.savefig("plots/05_actual_vs_pred.png", dpi=120); plt.show()

# =============================================================
# SECTION 8 - FEATURE IMPORTANCE
# =============================================================
print("\n" + "="*60)
print("SECTION 8 - FEATURE IMPORTANCE")
print("="*60)

if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=X.columns)
    imp = imp.sort_values(ascending=False)
    print(imp.to_string())

    plt.figure(figsize=(8,4))
    sns.barplot(x=imp.values, y=imp.index, palette="viridis")
    plt.title(f"Feature Importance ({best_name})")
    plt.tight_layout(); plt.savefig("plots/06_feat_importance.png", dpi=120); plt.show()
else:
    print("Best model has no feature_importances_ attribute.")

# =============================================================
# SECTION 9 - PREDICT NEW CAR
# =============================================================
print("\n" + "="*60)
print("SECTION 9 - SAMPLE PREDICTION")
print("="*60)

# Build a sample row using the same columns as X (excluding target)
sample = X.iloc[[0]].copy()
sample_pred = best_model.predict(sample)[0]
actual = y.iloc[0]
print("Sample input row:\n", sample.to_string(index=False))
print(f"\nPredicted Selling Price: {sample_pred:.2f}")
print(f"Actual Selling Price:    {actual:.2f}")

print("\n✅ Done! Plots saved in 'plots/' folder.")