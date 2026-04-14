

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import time

print("Loading preprocessed data from Step 2...")
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
print(f"✅ Loaded! Training on {len(X_train):,} samples.\n")


# Works like: if these features look suspicious → fraud
print("=" * 50)
print("MODEL 1: Training Logistic Regression...")
print("=" * 50)
start = time.time()

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

elapsed = time.time() - start
print(f"✅ Logistic Regression trained in {elapsed:.1f} seconds!")


print("\n" + "=" * 50)
print("MODEL 2: Training Random Forest (takes ~2-5 minutes)...")
print("=" * 50)
print("  💡 Why Random Forest?")
print("     - Combines 100 decision trees")
print("     - Each tree learns slightly different patterns")
print("     - They 'vote' on each prediction → more accurate!")
print("  Please wait...")

start = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,    # 100 decision trees
    random_state=42,     # for reproducibility
    n_jobs=-1,           # use all CPU cores to speed things up
    class_weight='balanced'  # extra care for fraud class
)
rf_model.fit(X_train, y_train)

elapsed = time.time() - start
print(f"✅ Random Forest trained in {elapsed:.1f} seconds!")

print("\n" + "=" * 50)
print("Saving trained models...")
print("=" * 50)

joblib.dump(lr_model, "logistic_regression_model.pkl")
joblib.dump(rf_model, "random_forest_model.pkl")
print("  ✅ Saved: logistic_regression_model.pkl")
print("  ✅ Saved: random_forest_model.pkl")


print("\n" + "=" * 50)
print("TOP 10 MOST IMPORTANT FEATURES (Random Forest):")
print("=" * 50)
feature_names = [f"V{i}" for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
importances = rf_model.feature_importances_
sorted_idx = importances.argsort()[::-1]

for i in range(10):
    idx = sorted_idx[i]
    print(f"  {i+1}. {feature_names[idx]:<16} importance = {importances[idx]:.4f}")

print("\n💡 These are the features the model relies on most to detect fraud.")

print("\n" + "=" * 50)
print("✅ STEP 3 COMPLETE!")
print("Next: Run  python step4_evaluate.py")
print("=" * 50)