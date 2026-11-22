# car-selling-price-prediction
# 🚗 Car Selling Price Prediction

An end-to-end Machine Learning project that predicts the **selling price of a used car** (in lakhs) based on its features — present/showroom price, kilometers driven, age, fuel type, seller type, transmission, and number of previous owners.

Built as part of an intermediate ML task using Python, scikit-learn, and a Tkinter GUI for predictions.

---

## 📌 Problem Statement

Buying or selling a used car is difficult without knowing its fair market value. This project builds a regression model trained on historical CarDekho data (301 rows, 9 features) that estimates a used car's selling price. Users enter car details into a simple desktop GUI and instantly get a predicted price.

---

## 📂 Project Structure

```
car-price-prediction/
├── car data.csv              # Dataset (301 rows, 9 columns)
├── analysis.py               # EDA + full training pipeline + plots
├── train_and_save.py         # Trains and saves model.pkl, encoders.pkl
├── gui.py                    # Tkinter desktop prediction app
├── model.pkl                 # Trained best model (Linear Regression)
├── encoders.pkl              # Label encoders for categorical features
├── feature_names.pkl         # Feature order used during training
├── plots/                    # EDA visualizations (auto-generated)
└── README.md
```

---

## 📊 Dataset

**Source:** CarDekho Used Car Dataset (public)

**Columns:**

| Column | Type | Description |
|---|---|---|
| Car_Name | string | Model name (dropped during training) |
| Year | int | Year of manufacture |
| Selling_Price | float | **Target** — selling price (in lakhs) |
| Present_Price | float | Showroom / current price (in lakhs) |
| Kms_Driven | int | Total kilometers driven |
| Fuel_Type | string | Petrol / Diesel / CNG |
| Seller_Type | string | Dealer / Individual |
| Transmission | string | Manual / Automatic |
| Owner | int | Number of previous owners (0–3) |

---

## 🛠️ Approach

### 1. Data Cleaning
- Standardized column names (lowercase, underscores)
- Removed 2 duplicate rows
- Confirmed no missing values

### 2. Feature Engineering
- **`car_age`** created as `CURRENT_YEAR (2024) − Year` — more meaningful than raw year
- Dropped `Car_Name` (98 unique values — too high cardinality for 301 rows)
- Label-encoded `Fuel_Type`, `Seller_Type`, `Transmission`

### 3. Modeling
Trained 3 regressors on an **80/20 train-test split**:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression** | **1.541** | **2.583** | **0.741** ✅ |
| Gradient Boosting | 1.205 | 2.680 | 0.721 |
| Random Forest | 1.467 | 3.480 | 0.530 |

**Best model:** Linear Regression (R² = 0.741)

> **Insight:** On a small dataset (301 rows), a simple linear model beats ensemble methods. Random Forest underperforms here because it overfits on limited data. This confirms that model complexity should be chosen based on data size, not just raw accuracy benchmarks.

### 4. Prediction Interface
A **Tkinter GUI** (`gui.py`) loads the saved model and lets users:
- Enter numeric inputs (price, km, age, owners)
- Select categories (fuel, seller, transmission) from dropdowns
- Click **Predict** → get an estimated price in lakhs

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 1. Explore & retrain (optional)
```bash
python analysis.py          # Runs EDA, saves plots to plots/
python train_and_save.py    # Retrains and overwrites model.pkl
```

### 2. Launch the prediction GUI
```bash
python gui.py
```

A window will open. Enter car details and click **Predict Price**.

### 3. Example Prediction
| Input | Value |
|---|---|
| Present Price | 8.0 lakhs |
| Kms Driven | 40000 |
| Car Age | 3 years |
| Previous Owners | 0 |
| Fuel Type | Diesel |
| Seller Type | Dealer |
| Transmission | Manual |
| **→ Predicted Price** | **~₹ 6.5 lakhs** |

---

## 📈 Visualizations

Generated into `plots/` by `analysis.py`:

- `01_target_dist.png` — Selling price distribution (right-skewed)
- `02_corr.png` — Correlation heatmap (Present_Price is strongest correlate)
- `03_box_*.png` — Selling price by fuel type, seller type, transmission
- `04_scatter.png` — Present price vs Selling price
- `05_actual_vs_pred.png` — Actual vs predicted scatter (best model)
- `06_feat_importance.png` — Feature importances

---

## 🔑 Key Insights

1. **Present Price** is the single strongest predictor of selling price (correlation ≈ 0.88).
2. **Car age** negatively impacts price — each extra year reduces value noticeably.
3. **Diesel cars** sell slightly higher than petrol on average.
4. **Dealer-sold cars** command higher prices than individual sellers.
5. **Manual transmission** dominates the dataset (85%+), but automatic cars sell at a premium when present.
6. **Cars with 0 previous owners** sell at noticeably higher prices.

---

## 🧰 Tech Stack

- **Python 3.11**
- **pandas**, **numpy** — data handling
- **scikit-learn** — modeling
- **matplotlib**, **seaborn** — EDA plots
- **Tkinter** — GUI (built into Python, no extra install)
- **pickle** — model persistence

---

## 👤 Author

Sankari  A
Car Selling Price Prediction

---

## 📄 License

This project is for educational purposes only.
