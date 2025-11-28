# 🏦 Hybrid Aksum Credit Risk Intelligence Engine

**An AI-powered B2B credit assessment system designed for Supply Chain & Procurement Finance.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Status](https://img.shields.io/badge/Status-Live-orange)

---

## ⚠️ Important Note on Data Confidentiality

**This repository does NOT contain real customer data.**

Due to strict Non-Disclosure Agreements (NDAs) and financial data privacy standards, all data used in this demonstration is **100% synthetic and algorithmically generated**. 

However, the data schema, feature engineering logic, and risk assessment architecture accurately depict the **production-grade methodology** required for assessing B2B credit risk. This project serves as a demonstration of the underlying engine logic without compromising sensitive financial information.

---

## 🎯 Project Overview

In the B2B supply chain sector (aligned with companies like **Aksum**), assessing the creditworthiness of SMEs and vendors is critical. Traditional scoring models are often black boxes.

This project builds a **Hybrid Risk Engine** that combines:
1.  **Quantitative ML:** XGBoost for statistical default prediction.
2.  **Qualitative Reasoning:** LLM (GenAI) for explaining decisions in plain English.
3.  **Historical Context:** Vector Search to find similar past cases.
4.  **Fraud Detection:** Anomaly detection for suspicious patterns.

## 🛠️ Tech Stack

| Component | Technology | Role |
|-----------|------------|------|
| **Core ML** | XGBoost | Calculating Default Probability (PD) |
| **Explainability** | SHAP (Shapley Values) | Explaining *why* a score is high/low |
| **Fraud** | Isolation Forest | Detecting anomalies in transaction patterns |
| **Vector DB** | FAISS | Retrieving similar historical customer profiles |
| **Reasoning** | Google Gemini API | Generating human-readable risk memos |
| **Backend** | FastAPI | Serving real-time credit decisions |
| **Frontend** | Streamlit | Interactive dashboard for credit analysts |

## 📊 Key Features

*   **Strict vs. Flexible Modes:** A/B testing capability for risk thresholds (Conservative vs. Growth focus).
*   **360° Risk View:** Analyzes Payment Delays, Credit Utilization, Dispute Rates, and Order Variance.
*   **Universal Assessment:** Capable of scoring *new* customers with zero history by using industry benchmarks.
*   **Automated Memos:** Generates email-ready approval/rejection drafts using GenAI.

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/aksum-credit-risk-engine.git
cd aksum-credit-risk-engine
2. Install Dependencies
Bash

pip install -r requirements.txt
3. Setup Environment
Create a .env file (if using LLM features):

text

GEMINI_API_KEY=your_api_key_here
4. Run the Dashboard
Bash

streamlit run dashboard.py
5. Run the API (Optional)
Bash

python run_api.py
📁 Project Structure
text

aksum-credit-risk-engine/
├── api/                 # FastAPI endpoints
├── data/                # Synthetic data generation logic
├── models/              # XGBoost & Fraud models
├── explainability/      # SHAP logic
├── vector_store/        # FAISS similarity search
├── llm_agent/           # GenAI integration
├── dashboard.py         # Streamlit UI
├── run_api.py           # Server entry point
└── README.md            # Documentation
📈 Model Logic (Synthetic Data)
The model is trained on synthetic B2B transaction patterns, including:

Payment Behavior: Average delay, max delay, late payment frequency.
Operational: Order consistency, lead time variance.
Financial: Credit utilization, dispute rates.
👨‍💻 Author
Manthan Teotia
Junior ML Engineer
Focus: Fintech, Risk Analytics, and B2B Supply Chain


