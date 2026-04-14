
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay
)

print("Loading test data and trained models...")
X_test  = np.load("X_test.npy")
y_test  = np.load("y_test.npy")
lr_model = joblib.load("logistic_regression_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
print("✅ Loaded!\n")



def evaluate_model(model, X_test, y_test, model_name):
    print("\n" + "=" * 55)
    print(f"  RESULTS: {model_name}")
    print("=" * 55)

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Confusion Matrix:")
    print(f"    True  Negatives  (Normal → Normal) : {tn:,}  ✅")
    print(f"    False Positives  (Normal → Fraud)  : {fp:,}  ❌ (false alarms)")
    print(f"    False Negatives  (Fraud  → Normal) : {fn:,}  ❌ (missed frauds!)")
    print(f"    True  Positives  (Fraud  → Fraud)  : {tp:,}  ✅ (caught!)")

    print(f"\n  Key Metrics:")
    fraud_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fraud_prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1           = 2 * fraud_prec * fraud_recall / (fraud_prec + fraud_recall) if (fraud_prec + fraud_recall) > 0 else 0
    roc_auc      = roc_auc_score(y_test, y_pred_prob)
    pr_auc       = average_precision_score(y_test, y_pred_prob)

    print(f"    Fraud Recall    : {fraud_recall*100:.1f}%  (caught {tp} of {tp+fn} frauds)")
    print(f"    Fraud Precision : {fraud_prec*100:.1f}%  (of fraud alerts, this many were real)")
    print(f"    F1 Score        : {f1:.4f}")
    print(f"    ROC-AUC Score   : {roc_auc:.4f}  (1.0 = perfect, 0.5 = random guess)")
    print(f"    PR-AUC Score    : {pr_auc:.4f}  (best metric for imbalanced data)")

    print(f"\n  Full Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    return y_pred, y_pred_prob, cm

lr_pred, lr_prob, lr_cm = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
rf_pred, rf_prob, rf_cm = evaluate_model(rf_model, X_test, y_test, "Random Forest")

print("\n📊 Generating evaluation charts...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Credit Card Fraud Detection - Model Evaluation", fontsize=15, fontweight='bold')


for ax, cm, title in zip(
    [axes[0, 0], axes[1, 0]],
    [lr_cm, rf_cm],
    ["Logistic Regression\nConfusion Matrix", "Random Forest\nConfusion Matrix"]
):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title)

for ax, prob, label, color in zip(
    [axes[0, 1], axes[1, 1]],
    [lr_prob, rf_prob],
    ["Logistic Regression", "Random Forest"],
    ["#2196F3", "#FF5722"]
):
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{label}\nROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


for ax, prob, label, color in zip(
    [axes[0, 2], axes[1, 2]],
    [lr_prob, rf_prob],
    ["Logistic Regression", "Random Forest"],
    ["#2196F3", "#FF5722"]
):
    precision, recall, _ = precision_recall_curve(y_test, prob)
    pr_auc = average_precision_score(y_test, prob)
    ax.plot(recall, precision, color=color, lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall (Fraud Caught %)")
    ax.set_ylabel("Precision")
    ax.set_title(f"{label}\nPrecision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("step4_evaluation.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart saved as 'step4_evaluation.png'")

