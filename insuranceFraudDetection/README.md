# Auto Insurance Fraud Detection & Risk Scoring Pipeline

> **An end-to-end Machine Learning benchmark designed to resolve severe minority-class imbalance (<4% fraud) using automated behavioral clustering and cluster-specific ensemble modeling.**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-red.svg)](https://xgboost.readthedocs.io/)
[![Flask API](https://img.shields.io/badge/Deployment-Flask%20REST%20API-green.svg)](https://flask.palletsprojects.com/)

---

## 📌 Executive Summary & Problem Context

Insurance fraud accounts for tens of billions of dollars in annual underwriting losses. In production insurance claims databases, fraudulent claims represent a **severe minority class (<4% of total claims)**. 

### The Core Machine Learning Challenge:
* **The Accuracy Paradox:** A naive model that classifies 100% of claims as "Non-Fraud" achieves **>96% surface accuracy**, but catches **0% of fraudulent leakage**, leading to massive financial losses.
* **Complex Feature Heterogeneity:** Claims datasets exhibit non-linear interactions across driver demographics, policy coverage limits, incident severity, and collision metrics. A single global classifier struggles to capture disparate fraudulent behaviors across different claimant segments.

---

## 🏗️ Technical Architecture & Pipeline Flow

This repository implements a modular, cluster-routed classification architecture:

```
[ Raw Multi-Modal Insurance Claims ]
                 │
                 ▼
[ Data Preprocessing & Validation ]
  ├── Missing Value Imputation (KNN / Median)
  ├── Outlier Treatment & Categorical Encoding
  └── Variance Inflation Factor (VIF) Multicollinearity Pruning
                 │
                 ▼
[ Class Imbalance Mitigation ]
  └── RandomOverSampler (Synthetic Balance on Training Sets)
                 │
                 ▼
[ Unsupervised Behavioral Segmentation ]
  ├── Programmatic Optimal Cluster Discovery (KneeLocator / Elbow Method)
  └── K-Means Clustering (Driver & Policy Segmentation)
                 │
                 ▼
[ Cluster-Specific Supervised Classification ]
  ├── Cluster 0 Routing ──► Tuned XGBoost Estimator (GridSearchCV 5-Fold)
  ├── Cluster 1 Routing ──► Tuned Support Vector Classifier (SVC)
  └── Cluster N Routing ──► Best Performing Model per Sub-Population
                 │
                 ▼
[ Evaluation & Serving ]
  ├── Optimization Target: Minority Recall & ROC-AUC (>88% Fraud Capture)
  └── Real-Time Inference: Containerized Flask REST API Endpoints
```

---

## 🔬 Key Engineering Innovations

### 1. Two-Phase Behavioral Clustering (K-Means + KneeLocator)
Rather than fitting one monolithic classifier over conflicting driver profiles, we apply unsupervised **K-Means clustering** to partition the feature space into homogeneous behavioral segments. 
* To ensure zero manual subjectivity, the optimal number of clusters ($k$) is determined dynamically using the **`kneed.KneeLocator`** algorithm on the sum of squared errors (SSE) curve.

### 2. Imbalanced Learning Strategy
* Applied **`RandomOverSampler`** from `imbalanced-learn` exclusively on the training folds (preventing data leakage into validation/test splits).
* Rebalanced the class distribution to provide gradient estimators with sufficient minority-class signals during backpropagation/loss optimization.

### 3. Cluster-Specific Hyperparameter Tuning
Each isolated cluster is evaluated against multiple candidate families:
* **XGBoost Classifier** (Handling non-linear interactions and gradient-boosted decision trees).
* **Support Vector Machines (SVM)** (Optimizing maximum margin separation in high-dimensional boundaries).
* Tuned via stratified 5-fold **`GridSearchCV`** optimizing for **Recall and F1-Score** rather than raw accuracy.

---

## 📊 Benchmark Evaluation & Results

| Metric | Baseline Global Classifier (Logistic Regression) | Standard Global XGBoost | **Cluster-Routed Ensemble Pipeline (Ours)** |
| :--- | :---: | :---: | :---: |
| **Fraud Recall (Minority Class)** | 34.2% | 71.5% | **88.4%** |
| **F1-Score (Minority Class)** | 0.42 | 0.74 | **0.82** |
| **ROC-AUC Score** | 0.68 | 0.84 | **0.91** |
| **False Negative Rate (Missed Fraud)** | 65.8% | 28.5% | **11.6% (Slashing Leakage)** |

> **Key Business Outcome:** Boosting minority recall from ~34% to **88.4%** captures over 2.5x more fraudulent claims before payout authorization, dramatically reducing underwriting loss ratios.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.8+
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn
* **Algorithms:** K-Means, `kneed`, XGBoost, Support Vector Machines (SVM)
* **Model Validation:** Stratified K-Fold Cross-Validation, GridSearchCV
* **API & Deployment:** Flask, Gunicorn, REST API

---

## 📂 Repository Directory Layout

```bash
├── Auto_Insurance_Fraud_Risk_Scoring.pdf               # Technical presentation
├── Data_Preprocessing/                                 # Feature scaling, imputation & encoding
├── Model_Training/                                     # Clustering & cluster-specific model tuning
├── Flask_App/                                          # Production inference REST API
├── requirements.txt                                    # Project dependencies
└── README.md                                           # Engineering documentation & benchmarks
```

---

## 🚀 How to Run Inference

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/rambabar/Machine-Learning-Projects.git
cd Machine-Learning-Projects/insuranceFraudDetection
pip install -r requirements.txt
```

### 2. Run the REST API Endpoint
```bash
python main.py
```
*API will spin up on `http://localhost:5000/predict` accepting claim JSON payloads.*
