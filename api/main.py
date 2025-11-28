# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

sys.path.append("..")

import pandas as pd

# create app
app = FastAPI(title="Aksum Credit Risk API")

# global models
credit_model = None
fraud_model = None
shap_explainer = None
case_retrieval = None
llm_agent = None
customer_data = None


class CustomerInput(BaseModel):
    avg_monthly_orders: float
    total_purchase_amount: float
    avg_order_value: float
    payment_delay_days_avg: float
    payment_delay_days_max: float
    credit_limit: float
    credit_utilization_pct: float
    num_invoices: int
    num_disputed_invoices: int
    dispute_rate: float
    days_since_first_order: int
    order_frequency_per_month: float
    lead_time_variance: float
    num_late_payments: int
    late_payment_rate: float


@app.on_event("startup")
async def startup():
    
    global credit_model, fraud_model, shap_explainer
    global case_retrieval, llm_agent, customer_data
    
    print("Loading models...")
    
    # import here to avoid path issues
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    from models.xgboost_model import AksumCreditModel
    from models.fraud_detector import AksumFraudDetector
    from explainability.shap_explainer import AksumExplainer
    from vector_store.case_retrieval import AksumCaseRetrieval
    from llm_agent.risk_reasoning import AksumLLMAgent
    import config
    
    # load credit model
    credit_model = AksumCreditModel()
    credit_model.load_model("saved_models/aksum_credit_model.pkl")
    
    # load fraud detector
    fraud_model = AksumFraudDetector()
    fraud_model.load_detector("saved_models")
    
    # load data
    customer_data = pd.read_csv("data/customer_data.csv")
    X = customer_data[config.FEATURE_NAMES]
    
    # setup explainer
    shap_explainer = AksumExplainer(credit_model.model)
    shap_explainer.setup_explainer(X)
    
    # setup retrieval
    case_retrieval = AksumCaseRetrieval()
    case_retrieval.load_index("vector_data")
    
    # setup llm
    llm_agent = AksumLLMAgent()
    
    print("All models loaded!")


@app.get("/")
async def root():
    return {"message": "Aksum Credit Risk API"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "xgboost": credit_model is not None,
            "fraud_detector": fraud_model is not None,
            "explainer": shap_explainer is not None,
            "retrieval": case_retrieval is not None,
            "llm_agent": llm_agent is not None
        }
    }


@app.post("/predict")
async def predict(customer: CustomerInput, mode: str = "strict"):
    
    # convert to dict
    data = {
        "avg_monthly_orders": customer.avg_monthly_orders,
        "total_purchase_amount": customer.total_purchase_amount,
        "avg_order_value": customer.avg_order_value,
        "payment_delay_days_avg": customer.payment_delay_days_avg,
        "payment_delay_days_max": customer.payment_delay_days_max,
        "credit_limit": customer.credit_limit,
        "credit_utilization_pct": customer.credit_utilization_pct,
        "num_invoices": customer.num_invoices,
        "num_disputed_invoices": customer.num_disputed_invoices,
        "dispute_rate": customer.dispute_rate,
        "days_since_first_order": customer.days_since_first_order,
        "order_frequency_per_month": customer.order_frequency_per_month,
        "lead_time_variance": customer.lead_time_variance,
        "num_late_payments": customer.num_late_payments,
        "late_payment_rate": customer.late_payment_rate
    }
    
    # get prediction
    result = credit_model.predict_single(data)
    
    # update category with mode
    cat = credit_model.get_risk_category(result["default_probability"], mode)
    result["risk_category"] = cat
    
    return {"prediction": result, "mode": mode}


@app.post("/fraud_check")
async def fraud_check(customer: CustomerInput):
    
    data = {
        "avg_monthly_orders": customer.avg_monthly_orders,
        "total_purchase_amount": customer.total_purchase_amount,
        "avg_order_value": customer.avg_order_value,
        "payment_delay_days_avg": customer.payment_delay_days_avg,
        "payment_delay_days_max": customer.payment_delay_days_max,
        "credit_limit": customer.credit_limit,
        "credit_utilization_pct": customer.credit_utilization_pct,
        "num_invoices": customer.num_invoices,
        "num_disputed_invoices": customer.num_disputed_invoices,
        "dispute_rate": customer.dispute_rate,
        "days_since_first_order": customer.days_since_first_order,
        "order_frequency_per_month": customer.order_frequency_per_month,
        "lead_time_variance": customer.lead_time_variance,
        "num_late_payments": customer.num_late_payments,
        "late_payment_rate": customer.late_payment_rate
    }
    
    result = fraud_model.detect_fraud(data)
    
    return {"fraud_analysis": result}


@app.post("/explain")
async def explain(customer: CustomerInput):
    
    data = {
        "avg_monthly_orders": customer.avg_monthly_orders,
        "total_purchase_amount": customer.total_purchase_amount,
        "avg_order_value": customer.avg_order_value,
        "payment_delay_days_avg": customer.payment_delay_days_avg,
        "payment_delay_days_max": customer.payment_delay_days_max,
        "credit_limit": customer.credit_limit,
        "credit_utilization_pct": customer.credit_utilization_pct,
        "num_invoices": customer.num_invoices,
        "num_disputed_invoices": customer.num_disputed_invoices,
        "dispute_rate": customer.dispute_rate,
        "days_since_first_order": customer.days_since_first_order,
        "order_frequency_per_month": customer.order_frequency_per_month,
        "lead_time_variance": customer.lead_time_variance,
        "num_late_payments": customer.num_late_payments,
        "late_payment_rate": customer.late_payment_rate
    }
    
    # prediction
    pred = credit_model.predict_single(data)
    
    # explanation
    exp = shap_explainer.explain_single(data)
    
    return {
        "prediction": pred,
        "explanation": {
            "base_risk": exp["base_risk_score"],
            "top_risk_factors": exp["top_3_risk_factors"],
            "top_positive_factors": exp["top_3_positive_factors"]
        }
    }


@app.post("/similar_cases")
async def similar_cases(customer: CustomerInput, num_cases: int = 3):
    
    data = {
        "avg_monthly_orders": customer.avg_monthly_orders,
        "total_purchase_amount": customer.total_purchase_amount,
        "avg_order_value": customer.avg_order_value,
        "payment_delay_days_avg": customer.payment_delay_days_avg,
        "payment_delay_days_max": customer.payment_delay_days_max,
        "credit_limit": customer.credit_limit,
        "credit_utilization_pct": customer.credit_utilization_pct,
        "num_invoices": customer.num_invoices,
        "num_disputed_invoices": customer.num_disputed_invoices,
        "dispute_rate": customer.dispute_rate,
        "days_since_first_order": customer.days_since_first_order,
        "order_frequency_per_month": customer.order_frequency_per_month,
        "lead_time_variance": customer.lead_time_variance,
        "num_late_payments": customer.num_late_payments,
        "late_payment_rate": customer.late_payment_rate
    }
    
    similar = case_retrieval.find_similar(data, num_results=num_cases)
    summary = case_retrieval.get_similar_summary(similar)
    
    return {
        "similar_cases": similar,
        "summary": summary
    }


@app.post("/compare_thresholds")
async def compare_thresholds(customer: CustomerInput):
    
    data = {
        "avg_monthly_orders": customer.avg_monthly_orders,
        "total_purchase_amount": customer.total_purchase_amount,
        "avg_order_value": customer.avg_order_value,
        "payment_delay_days_avg": customer.payment_delay_days_avg,
        "payment_delay_days_max": customer.payment_delay_days_max,
        "credit_limit": customer.credit_limit,
        "credit_utilization_pct": customer.credit_utilization_pct,
        "num_invoices": customer.num_invoices,
        "num_disputed_invoices": customer.num_disputed_invoices,
        "dispute_rate": customer.dispute_rate,
        "days_since_first_order": customer.days_since_first_order,
        "order_frequency_per_month": customer.order_frequency_per_month,
        "lead_time_variance": customer.lead_time_variance,
        "num_late_payments": customer.num_late_payments,
        "late_payment_rate": customer.late_payment_rate
    }
    
    pred = credit_model.predict_single(data)
    prob = pred["default_probability"]
    
    strict_cat = credit_model.get_risk_category(prob, "strict")
    flex_cat = credit_model.get_risk_category(prob, "flex")
    
    return {
        "default_probability": prob,
        "strict_mode": {"category": strict_cat},
        "flex_mode": {"category": flex_cat},
        "recommendation": "Both agree" if strict_cat == flex_cat else "Different"
    }


@app.get("/stats")
async def stats():
    
    llm_stats = llm_agent.get_api_stats()
    
    return {
        "llm_stats": llm_stats,
        "data_samples": len(customer_data) if customer_data is not None else 0
    }


@app.post("/full_analysis")
async def full_analysis(customer: CustomerInput):
    
    data = {
        "avg_monthly_orders": customer.avg_monthly_orders,
        "total_purchase_amount": customer.total_purchase_amount,
        "avg_order_value": customer.avg_order_value,
        "payment_delay_days_avg": customer.payment_delay_days_avg,
        "payment_delay_days_max": customer.payment_delay_days_max,
        "credit_limit": customer.credit_limit,
        "credit_utilization_pct": customer.credit_utilization_pct,
        "num_invoices": customer.num_invoices,
        "num_disputed_invoices": customer.num_disputed_invoices,
        "dispute_rate": customer.dispute_rate,
        "days_since_first_order": customer.days_since_first_order,
        "order_frequency_per_month": customer.order_frequency_per_month,
        "lead_time_variance": customer.lead_time_variance,
        "num_late_payments": customer.num_late_payments,
        "late_payment_rate": customer.late_payment_rate
    }
    
    # prediction
    pred = credit_model.predict_single(data)
    
    # fraud
    fraud = fraud_model.detect_fraud(data)
    
    # explanation
    exp = shap_explainer.explain_single(data)
    
    # similar cases
    similar = case_retrieval.find_similar(data, num_results=3)
    summary = case_retrieval.get_similar_summary(similar)
    
    # decision text
    decision = llm_agent.generate_decision_explanation(data, pred, exp, summary)
    
    return {
        "prediction": pred,
        "fraud_check": fraud,
        "explanation": {"top_risk_factors": exp["top_3_risk_factors"][:2]},
        "similar_cases": {"count": summary["num_similar_cases"], "default_rate": summary["default_rate_pct"]},
        "decision_text": decision
    }