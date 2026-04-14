

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

print("Loading data...")
df = pd.read_csv("../creditcard.csv")
print("✅ Data loaded!\n")


print("=" * 50)
print("STEP 2a: Scaling 'Amount' and 'Time' columns")
print("=" * 50)

scaler = StandardScaler()

df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_time']   = scaler.fit_transform(df[['Time']])

df.drop(['Amount', 'Time'], axis=1, inplace=True)

print("  ✅ 'Amount' and 'Time' scaled successfully!")
print(f"  scaled_amount range: {df['scaled_amount'].min():.2f} to {df['scaled_amount'].max():.2f}")
print(f"  scaled_time   range: {df['scaled_time'].min():.2f} to {df['scaled_time'].max():.2f}")

X = df.drop('Class', axis=1)
y = df['Class']

print(f"\n  Features (X) shape : {X.shape}")
print(f"  Labels   (y) shape : {y.shape}")
print(f"  Fraud in y         : {y.sum()} out of {len(y)}")


print("\n" + "=" * 50)
print("STEP 2b: Splitting data into Train and Test sets")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,          # 20% for testing, 80% for training
    random_state=42,        # 42 = seed for reproducibility (same split every time)
    stratify=y              # keeps the same fraud ratio in both sets
)

print(f"  Training set : {len(X_train):,} rows  ({y_train.sum()} frauds)")
print(f"  Testing  set : {len(X_test):,} rows  ({y_test.sum()} frauds)")

print("\n" + "=" * 50)
print("STEP 2c: Applying SMOTE to balance the training data")
print("=" * 50)
print("  (This may take 1-2 minutes...)")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\n  BEFORE SMOTE:")
print(f"    Normal : {(y_train == 0).sum():,}")
print(f"    Fraud  : {(y_train == 1).sum():,}")
print(f"\n  AFTER SMOTE:")
print(f"    Normal : {(y_train_balanced == 0).sum():,}")
print(f"    Fraud  : {(y_train_balanced == 1).sum():,}")
print("  ✅ Now balanced! The model can learn fraud patterns properly.")


print("\n" + "=" * 50)
print("STEP 2d: Saving processed data...")
print("=" * 50)

np.save("X_train.npy", X_train_balanced)
np.save("y_train.npy", y_train_balanced)
np.save("X_test.npy",  X_test)
np.save("y_test.npy",  y_test)

print("  ✅ Saved: X_train.npy, y_train.npy, X_test.npy, y_test.npy")

