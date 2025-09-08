# 📊 Payment Default / Credit Risk Prediction  
*Aksum.co.in — End-to-End Machine Learning Project*

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)  
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)  
![Status](https://img.shields.io/badge/Project%20Stage-Complete-brightgreen.svg)  
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 🌟 Overview  

This project by **Aksum.co.in** demonstrates how **Machine Learning** can be applied to **predict client payment delays/defaults** in a B2B trading environment.  
It’s a complete pipeline — from **EDA** and **feature engineering**, to **model training**, **evaluation**, and a **professional Streamlit app** for business teams.  

The goal: **reduce risk exposure, negotiate better terms, and protect working capital**.  

---

## 🚀 Key Features  

✔️ **Exploratory Data Analysis (EDA)**  
- Delay patterns by client, material type, payment method, invoice amount.  

✔️ **Feature Engineering**  
- Calendar features: month, quarter, due day of week, month-end flag.  
- Historical features: prior delay %, avg delay days, rolling 90-day spend.  
- Client-relative stats: z-score of invoice amount.  
- One-hot encoding of payment method and material type.  

✔️ **Modeling**  
- Baseline: Logistic Regression.  
- Production: Random Forest.  
- Experimental: XGBoost.  
- Threshold tuning with Precision-Recall tradeoff.  

✔️ **Explainability**  
- Feature importances.  
- Invoice-level “reason codes” (heuristic top drivers).  

✔️ **Deployment**  
- **Streamlit App** with two flows:  
  1. **Processed features → Score**  
  2. **Raw invoices → Auto feature build → Score** (business self-serve)  
- Downloadable templates & results.  
- Branded, corporate-looking dashboard.  


## 📂 Repository Structure  

credit-risk/
├── .streamlit/ <- UI theme config
├── assets/ <- Logos & branding
├── data/ <- Raw, interim, processed data
├── models/ <- Trained model & metadata
├── notebooks/ <- EDA, feature build, modeling
├── reports/ <- Figures & business readouts
├── src/
│ ├── app/ <- Streamlit app
│ ├── features/ <- Online builder (raw → features)
│ └── modeling/ <- Model training scripts
├── requirements.txt <- Python dependencies
└── README.md <- Project documentation

