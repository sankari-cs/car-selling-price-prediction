# gui.py — lightweight car price predictor (no extra libraries needed)
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

with open("model.pkl", "rb") as f: model = pickle.load(f)
with open("encoders.pkl", "rb") as f: encoders = pickle.load(f)
with open("feature_names.pkl", "rb") as f: feature_names = pickle.load(f)

root = tk.Tk()
root.title("🚗 Car Selling Price Predictor")
root.geometry("420x520")
root.configure(padx=20, pady=20)

tk.Label(root, text="Car Selling Price Predictor",
         font=("Arial", 16, "bold")).pack(pady=10)

def make_field(label, default):
    tk.Label(root, text=label, anchor="w").pack(fill="x", pady=(6, 0))
    e = tk.Entry(root)
    e.insert(0, str(default))
    e.pack(fill="x")
    return e

e_present = make_field("Present Price (lakhs)", 5.0)
e_kms     = make_field("Kms Driven", 30000)
e_age     = make_field("Car Age (years)", 5)
e_owner   = make_field("Previous Owners (0-3)", 0)

tk.Label(root, text="Fuel Type", anchor="w").pack(fill="x", pady=(6, 0))
cb_fuel = ttk.Combobox(root, values=list(encoders["fuel_type"].classes_), state="readonly")
cb_fuel.current(0); cb_fuel.pack(fill="x")

tk.Label(root, text="Seller Type", anchor="w").pack(fill="x", pady=(6, 0))
cb_seller = ttk.Combobox(root, values=list(encoders["seller_type"].classes_), state="readonly")
cb_seller.current(0); cb_seller.pack(fill="x")

tk.Label(root, text="Transmission", anchor="w").pack(fill="x", pady=(6, 0))
cb_trans = ttk.Combobox(root, values=list(encoders["transmission"].classes_), state="readonly")
cb_trans.current(0); cb_trans.pack(fill="x")

result = tk.Label(root, text="", font=("Arial", 14, "bold"), fg="green")
result.pack(pady=20)

def predict():
    try:
        row = {
            "present_price": float(e_present.get()),
            "kms_driven":    int(e_kms.get()),
            "fuel_type":     encoders["fuel_type"].transform([cb_fuel.get()])[0],
            "seller_type":   encoders["seller_type"].transform([cb_seller.get()])[0],
            "transmission":  encoders["transmission"].transform([cb_trans.get()])[0],
            "owner":         int(e_owner.get()),
            "car_age":       int(e_age.get()),
        }
        X_in = pd.DataFrame([row])[feature_names]
        price = model.predict(X_in)[0]
        result.config(text=f"Estimated Price: ₹ {price:.2f} lakhs")
    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Button(root, text="💰 Predict Price", command=predict,
          bg="#4CAF50", fg="white", font=("Arial", 12, "bold")).pack(pady=10, fill="x")

root.mainloop()