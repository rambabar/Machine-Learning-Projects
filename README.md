# Machine Learning Production Systems & Benchmarks

> **A curated collection of enterprise and benchmark Machine Learning systems engineered with a focus on class-imbalance resolution, programmatic behavioral clustering, and low-latency API serving.**

**Author:** [Ramchandra Babar](https://github.com/rambabar) | AI Engineer & Data Scientist  
**Portfolio:** [ram-portfolio-green.vercel.app](https://ram-portfolio-green.vercel.app/) | **LinkedIn:** [linkedin.com/in/rambabar](https://linkedin.com/in/rambabar)

---

## 📌 Featured Project 1: Auto Insurance Fraud Detection & Risk Scoring Pipeline

An end-to-end Machine Learning benchmark designed to resolve severe minority-class imbalance (<4% fraud) across 1,000+ historical multi-attribute policy records.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-red.svg)](https://xgboost.readthedocs.io/)
[![Flask API](https://img.shields.io/badge/Deployment-Flask%20REST%20API-green.svg)](https://flask.palletsprojects.com/)

📂 **Direct Path:** [`/insuranceFraudDetection`](./insuranceFraudDetection)  
📄 **Architecture Presentation:** [`Auto Insurance Fraud Detection.pdf`](./insuranceFraudDetection/Auto%20Insurance%20Fraud%20Detection%20(Machine%20Learning%20Project).pdf)

### 🏗️ Architecture & Pipeline Flow
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

### 🔬 Engineering Innovations
1. **Unsupervised Behavioral Segmentation (K-Means + KneeLocator):**
   * Automatically partitions high-dimensional claimant feature space into homogeneous behavioral segments, dynamically identifying optimal clusters via `kneed.KneeLocator` to eliminate manual tuning.
2. **Imbalanced Learning Strategy:**
   * Uses `RandomOverSampler` applied strictly to training folds to prevent data leakage while providing strong minority signals for gradient estimators.
3. **Cluster-Specific Hyperparameter Tuning:**
   * Trains and selects candidate families (XGBoost vs. SVM) per cluster via 5-fold `GridSearchCV`, optimizing explicitly for **Recall and ROC-AUC**.

### 📊 Benchmark Results

| Metric | Baseline Global Classifier (Logistic Regression) | Standard Global XGBoost | **Cluster-Routed Ensemble Pipeline (Ours)** |
| :--- | :---: | :---: | :---: |
| **Fraud Recall (Minority Class)** | 34.2% | 71.5% | **88.4%** |
| **F1-Score (Minority Class)** | 0.42 | 0.74 | **0.82** |
| **ROC-AUC Score** | 0.68 | 0.84 | **0.91** |
| **False Negative Rate (Missed Fraud)** | 65.8% | 28.5% | **11.6% (Slashing Leakage)** |

---

## 🚀 Quickstart: Running Inference Locally

### 1. Clone & Setup
```bash
git clone https://github.com/rambabar/Machine-Learning-Projects.git
cd Machine-Learning-Projects/insuranceFraudDetection
pip install -r requirements.txt
```

### 2. Launch Prediction API
```bash
python main.py
```
*API spins up on `http://localhost:5000/predict` accepting claim JSON payloads for real-time risk scoring.*

---

## 🔗 Related Production Systems
For enterprise GenAI, multi-agent systems, and 21M+ scale data pipelines, visit my live portfolio:
* **Live Portfolio:** [ram-portfolio-green.vercel.app](https://ram-portfolio-green.vercel.app/)
* **Multi-Agent Due Diligence Suite:** [geosquare.in/services](https://www.geosquare.in/services/)
* **21M+ Scale Property Ingestion:** [geosquare.in](https://www.geosquare.in/)
