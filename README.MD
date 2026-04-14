# Hi, I'm Darshil Gandhi 👋

I'm a student at the **University of South Florida** passionate about **Data Science in Financial Services**. I love using machine learning to solve real-world problems in finance — fraud detection, risk modelling, and everything in between.

---

## 🚀 Featured Project — Credit Card Fraud Detection

### The Problem
Credit card fraud costs billions of dollars every year. Banks process millions of transactions daily and need an automated way to flag suspicious activity in real time — without blocking legitimate customers.

### What I Built
A machine learning pipeline that analyses credit card transactions and predicts whether they are fraudulent or legitimate.

**Dataset:** 284,807 real European credit card transactions (September 2013) — only 0.17% are fraud, making this a challenging imbalanced classification problem.

### My Approach
- **Explored** the dataset to understand the class imbalance (492 frauds vs 284,315 normal)
- **Scaled** the Amount and Time features using StandardScaler so all features are on the same range
- **Balanced** the training data using SMOTE (Synthetic Minority Over-sampling Technique) to prevent the model from ignoring fraud cases
- **Trained** two models — Logistic Regression and Random Forest — and compared their performance
- **Evaluated** using Precision, Recall, F1 Score, and ROC-AUC (the right metrics for imbalanced data)

### Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

| Library | Used for |
|---------|----------|
| `pandas` | Loading and exploring the dataset |
| `scikit-learn` | StandardScaler, train/test split, Random Forest, Logistic Regression, evaluation metrics |
| `imbalanced-learn` | SMOTE to handle class imbalance |
| `matplotlib / seaborn` | Visualisations and charts |
| `joblib` | Saving and loading trained models |

### Key Results
- Random Forest outperformed Logistic Regression on fraud recall
- SMOTE significantly improved the model's ability to catch rare fraud cases
- Evaluated on held-out test data the model never saw during training

### What I Learned
- How to handle highly imbalanced datasets in real-world scenarios
- Why accuracy alone is a misleading metric for fraud detection
- The difference between Precision and Recall and why Recall matters more here
- How Random Forest uses ensemble learning (100 decision trees voting together)

---

## 📫 Connect with Me
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/darshilgandhi)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:darshilgandhi20@gmail.com)

---

*Currently learning: Machine Learning • Data Science • Python • Financial Analytics*
