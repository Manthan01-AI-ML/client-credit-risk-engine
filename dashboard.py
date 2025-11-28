

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import time


sys.path.append("models")
sys.path.append("explainability")
sys.path.append("vector_store")
sys.path.append("llm_agent")


st.set_page_config(
    page_title="Aksum Credit Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom css
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #1a1a2e 100%);
    }
    
    h1 {
        color: #00d4ff !important;
        font-weight: 700 !important;
        text-align: center;
        padding: 20px 0;
    }
    
    h2 {
        color: #e94560 !important;
        font-weight: 600 !important;
    }
    
    h3 {
        color: #00d4ff !important;
    }
    
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1f4068 0%, #162447 100%);
        border: 1px solid #00d4ff;
        border-radius: 15px;
        padding: 20px;
    }
    
    [data-testid="metric-container"] label {
        color: #a0a0a0 !important;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #00d4ff !important;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #e94560 0%, #0f3460 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 40px;
        font-size: 18px;
        font-weight: 600;
    }
    
    .stNumberInput input {
        background: #162447 !important;
        color: white !important;
        border: 1px solid #00d4ff !important;
        border-radius: 10px !important;
    }
    
    .stTextInput input {
        background: #162447 !important;
        color: white !important;
        border: 1px solid #00d4ff !important;
        border-radius: 10px !important;
    }
    
    .stSelectbox > div > div {
        background: #162447 !important;
        color: white !important;
        border: 1px solid #00d4ff !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)


# load models
@st.cache_resource
def load_models():
    
    from models.xgboost_model import AksumCreditModel
    from models.fraud_detector import AksumFraudDetector
    from explainability.shap_explainer import AksumExplainer
    from vector_store.case_retrieval import AksumCaseRetrieval
    from llm_agent.risk_reasoning import AksumLLMAgent
    import config
    
    model = AksumCreditModel()
    model.load_model("saved_models/aksum_credit_model.pkl")
    
    fraud = AksumFraudDetector()
    fraud.load_detector("saved_models")
    
    data = pd.read_csv("data/customer_data.csv")
    X = data[config.FEATURE_NAMES]
    
    explainer = AksumExplainer(model.model)
    explainer.setup_explainer(X)
    
    retrieval = AksumCaseRetrieval()
    retrieval.load_index("vector_data")
    
    llm = AksumLLMAgent()
    
    return model, fraud, explainer, retrieval, llm


model, fraud_detector, explainer, retrieval, llm_agent = load_models()


# sidebar
with st.sidebar:
    
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #00d4ff; font-size: 28px; margin: 0;'>🏦 AKSUM</h1>
        <p style='color: #e94560; font-size: 14px;'>Credit Risk Assessment</p>
        <p style='color: #666; font-size: 11px;'>Universal B2B Tool</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "📌 Select Service",
        [
            "🏠 Home",
            "📝 New Customer Assessment",
            "📊 Quick Assessment",
            "📈 Risk Calculator",
            "ℹ️ How It Works"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Settings")
    
    threshold_mode = st.selectbox(
        "Assessment Mode",
        ["Standard (Balanced)", "Conservative (Strict)", "Growth (Flexible)"],
        help="Standard: Normal assessment | Conservative: Stricter rules | Growth: Relaxed rules"
    )
    
    if "Conservative" in threshold_mode:
        mode = "strict"
    elif "Growth" in threshold_mode:
        mode = "flex"
    else:
        mode = "strict"
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center;'>
        <p style='color: #666; font-size: 11px;'>Powered by</p>
        <a href='https://aksum.co.in' target='_blank' style='color: #00d4ff; text-decoration: none;'>
            aksum.co.in
        </a>
    </div>
    """, unsafe_allow_html=True)


# home page
if page == "🏠 Home":
    
    st.markdown("""
    <div style='text-align: center; padding: 40px 0;'>
        <h1 style='font-size: 42px;'>🏦 B2B Credit Risk Assessment</h1>
        <p style='color: #a0a0a0; font-size: 20px;'>Check creditworthiness of any business customer</p>
        <p style='color: #666; font-size: 14px;'>No signup required • Instant results • Free to use</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # what we offer
    st.markdown("### 🎯 What You Can Do")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1f4068 0%, #162447 100%); 
                    padding: 25px; border-radius: 15px; text-align: center;
                    border: 1px solid #00d4ff; height: 200px;'>
            <h1 style='font-size: 40px; margin: 0;'>📝</h1>
            <h3 style='color: #00d4ff; margin: 10px 0;'>New Customer</h3>
            <p style='color: #a0a0a0; font-size: 13px;'>Full assessment for new business customer with all details</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1f4068 0%, #162447 100%); 
                    padding: 25px; border-radius: 15px; text-align: center;
                    border: 1px solid #e94560; height: 200px;'>
            <h1 style='font-size: 40px; margin: 0;'>⚡</h1>
            <h3 style='color: #e94560; margin: 10px 0;'>Quick Check</h3>
            <p style='color: #a0a0a0; font-size: 13px;'>Fast assessment with minimal inputs for quick decision</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1f4068 0%, #162447 100%); 
                    padding: 25px; border-radius: 15px; text-align: center;
                    border: 1px solid #ffbb00; height: 200px;'>
            <h1 style='font-size: 40px; margin: 0;'>🧮</h1>
            <h3 style='color: #ffbb00; margin: 10px 0;'>Risk Calculator</h3>
            <p style='color: #a0a0a0; font-size: 13px;'>Calculate risk score based on key parameters</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1f4068 0%, #162447 100%); 
                    padding: 25px; border-radius: 15px; text-align: center;
                    border: 1px solid #00ff88; height: 200px;'>
            <h1 style='font-size: 40px; margin: 0;'>🔍</h1>
            <h3 style='color: #00ff88; margin: 10px 0;'>Explainable</h3>
            <p style='color: #a0a0a0; font-size: 13px;'>Understand why customer is risky or safe</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # how it works
    st.markdown("### 🔄 How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div style='background: #e94560; width: 50px; height: 50px; border-radius: 50%; 
                        margin: 0 auto 15px; display: flex; align-items: center; justify-content: center;'>
                <span style='color: white; font-size: 24px; font-weight: bold;'>1</span>
            </div>
            <h4 style='color: white;'>Enter Details</h4>
            <p style='color: #666; font-size: 12px;'>Input customer business information</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div style='background: #e94560; width: 50px; height: 50px; border-radius: 50%; 
                        margin: 0 auto 15px; display: flex; align-items: center; justify-content: center;'>
                <span style='color: white; font-size: 24px; font-weight: bold;'>2</span>
            </div>
            <h4 style='color: white;'>AI Analysis</h4>
            <p style='color: #666; font-size: 12px;'>Our ML model analyzes the data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div style='background: #e94560; width: 50px; height: 50px; border-radius: 50%; 
                        margin: 0 auto 15px; display: flex; align-items: center; justify-content: center;'>
                <span style='color: white; font-size: 24px; font-weight: bold;'>3</span>
            </div>
            <h4 style='color: white;'>Get Score</h4>
            <p style='color: #666; font-size: 12px;'>Receive risk score and category</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div style='background: #e94560; width: 50px; height: 50px; border-radius: 50%; 
                        margin: 0 auto 15px; display: flex; align-items: center; justify-content: center;'>
                <span style='color: white; font-size: 24px; font-weight: bold;'>4</span>
            </div>
            <h4 style='color: white;'>Take Action</h4>
            <p style='color: #666; font-size: 12px;'>Make informed credit decision</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # cta
    st.markdown("""
    <div style='text-align: center; padding: 30px 0;'>
        <p style='color: #a0a0a0; font-size: 16px;'>Ready to assess a customer?</p>
        <p style='color: #00d4ff; font-size: 14px;'>👈 Select "New Customer Assessment" from sidebar</p>
    </div>
    """, unsafe_allow_html=True)


# new customer assessment page
elif page == "📝 New Customer Assessment":
    
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1>📝 New Customer Assessment</h1>
        <p style='color: #a0a0a0;'>Enter business customer details for complete credit evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # customer info section
    st.markdown("### 👤 Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        customer_name = st.text_input(
            "Business Name",
            value="",
            placeholder="Enter business name",
            help="Name of the business customer"
        )
    
    with col2:
        customer_id = st.text_input(
            "Customer ID (Optional)",
            value="",
            placeholder="Your internal ID",
            help="Your reference ID for this customer"
        )
    
    with col3:
        industry = st.selectbox(
            "Industry",
            ["Select Industry", "Manufacturing", "Retail", "Wholesale", "Services", "Trading", "Other"],
            help="Business industry type"
        )
    
    st.markdown("---")
    
    # business data
    st.markdown("### 📊 Business Data")
    
    tab1, tab2, tab3 = st.tabs(["📦 Order History", "💳 Credit Profile", "💰 Payment History"])
    
    with tab1:
        st.markdown("**Enter order and purchase information:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_monthly_orders = st.number_input(
                "Average Monthly Orders",
                value=0.0,
                min_value=0.0,
                max_value=500.0,
                step=1.0,
                help="How many orders per month on average?"
            )
        
        with col2:
            total_purchase = st.number_input(
                "Total Purchase Amount (₹)",
                value=0.0,
                min_value=0.0,
                max_value=100000000.0,
                step=10000.0,
                help="Total business done in rupees"
            )
        
        with col3:
            avg_order_value = st.number_input(
                "Average Order Value (₹)",
                value=0.0,
                min_value=0.0,
                max_value=10000000.0,
                step=1000.0,
                help="Average value per order"
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            days_since_first = st.number_input(
                "Relationship Duration (Days)",
                value=0,
                min_value=0,
                max_value=3650,
                help="How long has this customer been with you?"
            )
        
        with col2:
            order_freq = st.number_input(
                "Order Frequency per Month",
                value=0.0,
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                help="How often they place orders"
            )
        
        with col3:
            lead_time_var = st.number_input(
                "Lead Time Variance (Days)",
                value=0.0,
                min_value=0.0,
                max_value=60.0,
                step=0.5,
                help="Variation in delivery acceptance"
            )
    
    with tab2:
        st.markdown("**Enter credit and invoice information:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            credit_limit = st.number_input(
                "Credit Limit (₹)",
                value=0.0,
                min_value=0.0,
                max_value=100000000.0,
                step=10000.0,
                help="Credit limit you want to give or have given"
            )
        
        with col2:
            credit_util = st.number_input(
                "Credit Utilization (%)",
                value=0.0,
                min_value=0.0,
                max_value=200.0,
                step=1.0,
                help="How much of credit limit is being used? Can be over 100% if overdue"
            )
        
        with col3:
            num_invoices = st.number_input(
                "Total Invoices",
                value=0,
                min_value=0,
                max_value=10000,
                help="Total number of invoices raised"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_disputed = st.number_input(
                "Disputed Invoices",
                value=0,
                min_value=0,
                max_value=500,
                help="Number of invoices with disputes"
            )
        
        with col2:
            if num_invoices > 0:
                dispute_rate = round((num_disputed / num_invoices) * 100, 2)
            else:
                dispute_rate = 0.0
            
            st.metric("Dispute Rate", f"{dispute_rate}%")
            st.caption("Auto-calculated from above")
    
    with tab3:
        st.markdown("**Enter payment behavior information:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            payment_delay_avg = st.number_input(
                "Average Payment Delay (Days)",
                value=0.0,
                min_value=0.0,
                max_value=365.0,
                step=1.0,
                help="On average, how many days late are payments?"
            )
        
        with col2:
            payment_delay_max = st.number_input(
                "Maximum Payment Delay (Days)",
                value=0.0,
                min_value=0.0,
                max_value=365.0,
                step=1.0,
                help="Worst case - maximum days late ever"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_late = st.number_input(
                "Number of Late Payments",
                value=0,
                min_value=0,
                max_value=500,
                help="Total count of late payments"
            )
        
        with col2:
            if num_invoices > 0:
                late_rate = round((num_late / num_invoices) * 100, 2)
            else:
                late_rate = 0.0
            
            st.metric("Late Payment Rate", f"{late_rate}%")
            st.caption("Auto-calculated from above")
    
    st.markdown("---")
    
    # validation
    all_filled = (
        avg_monthly_orders > 0 or
        total_purchase > 0 or
        credit_limit > 0 or
        num_invoices > 0
    )
    
    # submit button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if all_filled:
            assess_clicked = st.button("🔍 ASSESS CREDIT RISK", use_container_width=True)
        else:
            st.warning("⚠️ Please enter at least some business data to proceed")
            assess_clicked = False
    
    if assess_clicked:
        
        # progress
        progress = st.progress(0)
        status = st.empty()
        
        status.text("📊 Processing customer data...")
        progress.progress(20)
        time.sleep(0.3)
        
        # create customer dict
        customer = {
            "avg_monthly_orders": avg_monthly_orders if avg_monthly_orders > 0 else 1.0,
            "total_purchase_amount": total_purchase if total_purchase > 0 else 10000.0,
            "avg_order_value": avg_order_value if avg_order_value > 0 else 1000.0,
            "payment_delay_days_avg": payment_delay_avg,
            "payment_delay_days_max": payment_delay_max if payment_delay_max >= payment_delay_avg else payment_delay_avg,
            "credit_limit": credit_limit if credit_limit > 0 else 100000.0,
            "credit_utilization_pct": credit_util,
            "num_invoices": num_invoices if num_invoices > 0 else 1,
            "num_disputed_invoices": num_disputed,
            "dispute_rate": dispute_rate,
            "days_since_first_order": days_since_first if days_since_first > 0 else 30,
            "order_frequency_per_month": order_freq if order_freq > 0 else 1.0,
            "lead_time_variance": lead_time_var,
            "num_late_payments": num_late,
            "late_payment_rate": late_rate,
        }
        
        status.text("🤖 Running AI analysis...")
        progress.progress(40)
        time.sleep(0.3)
        
        # predict
        prediction = model.predict_single(customer)
        cat = model.get_risk_category(prediction["default_probability"], mode)
        prediction["risk_category"] = cat
        
        status.text("🛡️ Checking fraud patterns...")
        progress.progress(60)
        time.sleep(0.3)
        
        # fraud
        fraud_result = fraud_detector.detect_fraud(customer)
        
        status.text("🔍 Analyzing risk factors...")
        progress.progress(80)
        time.sleep(0.3)
        
        # explain
        exp_result = explainer.explain_single(customer)
        
        # similar cases
        similar = retrieval.find_similar(customer, num_results=5)
        summary = retrieval.get_similar_summary(similar)
        
        status.text("📝 Generating recommendation...")
        progress.progress(100)
        time.sleep(0.3)
        
        # decision
        decision = llm_agent.generate_decision_explanation(
            customer, prediction, exp_result, summary
        )
        
        progress.empty()
        status.empty()
        
        st.markdown("---")
        
        # results
        prob_pct = round(prediction["default_probability"] * 100, 1)
        risk_cat = prediction["risk_category"]
        
        # colors
        if risk_cat == "LOW":
            main_color = "#00ff88"
            bg_color = "rgba(0, 255, 136, 0.1)"
            status_emoji = "✅"
            status_text = "LOW RISK - APPROVE"
            recommendation = "Safe to extend credit"
        elif risk_cat == "MEDIUM":
            main_color = "#ffbb00"
            bg_color = "rgba(255, 187, 0, 0.1)"
            status_emoji = "⚠️"
            status_text = "MEDIUM RISK - CONDITIONAL"
            recommendation = "Approve with monitoring"
        elif risk_cat == "HIGH":
            main_color = "#ff6b35"
            bg_color = "rgba(255, 107, 53, 0.1)"
            status_emoji = "🔶"
            status_text = "HIGH RISK - RESTRICTED"
            recommendation = "Reduce credit or add guarantees"
        else:
            main_color = "#ff0055"
            bg_color = "rgba(255, 0, 85, 0.1)"
            status_emoji = "❌"
            status_text = "VERY HIGH RISK - DECLINE"
            recommendation = "Do not extend credit"
        
        # result header
        display_name = customer_name if customer_name else "Customer"
        
        st.markdown(f"""
        <div style='background: {bg_color}; border: 2px solid {main_color}; 
                    border-radius: 20px; padding: 30px; text-align: center; margin: 20px 0;'>
            <p style='color: #a0a0a0; font-size: 14px; margin: 0;'>Assessment Result for</p>
            <h2 style='color: white; margin: 5px 0;'>{display_name}</h2>
            <h1 style='font-size: 60px; margin: 10px 0;'>{status_emoji}</h1>
            <h2 style='color: {main_color}; font-size: 28px; margin: 10px 0;'>{status_text}</h2>
            <p style='color: #a0a0a0; font-size: 16px;'>{recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Default Risk", f"{prob_pct}%")
        
        with col2:
            st.metric("Risk Category", risk_cat)
        
        with col3:
            st.metric("Fraud Risk", fraud_result["fraud_risk_level"])
        
        with col4:
            st.metric("Benchmark Default Rate", f"{summary['default_rate_pct']}%")
        
        st.markdown("---")
        
        # gauge and factors
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Risk Score")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                number={"suffix": "%", "font": {"size": 40, "color": "white"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": main_color},
                    "bgcolor": "#162447",
                    "steps": [
                        {"range": [0, 30], "color": "rgba(0, 255, 136, 0.3)"},
                        {"range": [30, 50], "color": "rgba(255, 187, 0, 0.3)"},
                        {"range": [50, 70], "color": "rgba(255, 107, 53, 0.3)"},
                        {"range": [70, 100], "color": "rgba(255, 0, 85, 0.3)"}
                    ]
                }
            ))
            
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=300,
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Key Risk Factors")
            
            risk_factors = exp_result["top_3_risk_factors"]
            pos_factors = exp_result["top_3_positive_factors"]
            
            st.markdown("**⚠️ Factors Increasing Risk:**")
            
            if len(risk_factors) > 0:
                for factor in risk_factors[:3]:
                    name = factor["feature"].replace("_", " ").title()
                    value = factor["value"]
                    st.markdown(f"""
                    <div style='background: rgba(255, 0, 85, 0.1); border-left: 3px solid #ff0055;
                                padding: 10px 15px; margin: 5px 0; border-radius: 5px;'>
                        <span style='color: #ff6b6b;'>📍 {name}</span>
                        <span style='color: white; float: right;'>{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("*No major risk factors identified*")
            
            st.markdown("")
            st.markdown("**✅ Factors Reducing Risk:**")
            
            if len(pos_factors) > 0:
                for factor in pos_factors[:3]:
                    name = factor["feature"].replace("_", " ").title()
                    value = factor["value"]
                    st.markdown(f"""
                    <div style='background: rgba(0, 255, 136, 0.1); border-left: 3px solid #00ff88;
                                padding: 10px 15px; margin: 5px 0; border-radius: 5px;'>
                        <span style='color: #00ff88;'>📍 {name}</span>
                        <span style='color: white; float: right;'>{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("*No significant positive factors*")
        
        st.markdown("---")
        
        # recommendation
        st.markdown("### 📋 Credit Recommendation")
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1f4068 0%, #162447 100%);
                    border: 1px solid {main_color}; border-radius: 15px; padding: 25px;'>
            <pre style='color: white; white-space: pre-wrap; font-family: sans-serif; margin: 0; font-size: 14px;'>{decision}</pre>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # benchmark info
        with st.expander("📊 Benchmark Comparison (How this compares to similar businesses)"):
            
            st.markdown(f"""
            Based on our database of **{summary['num_similar_cases']} similar business profiles**:
            
            - **{summary['num_defaulted']}** similar businesses defaulted in the past
            - **{summary['num_good']}** similar businesses remained in good standing
            - Historical default rate for this profile type: **{summary['default_rate_pct']}%**
            
            *This benchmark is based on anonymized industry data patterns.*
            """)


# quick assessment page
elif page == "📊 Quick Assessment":
    
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1>⚡ Quick Credit Check</h1>
        <p style='color: #a0a0a0;'>Fast assessment with just 4 key parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Enter Key Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        quick_delay = st.slider(
            "📅 Average Payment Delay (Days)",
            min_value=0,
            max_value=90,
            value=15,
            help="How many days late are payments on average?"
        )
        
        quick_util = st.slider(
            "💳 Credit Utilization (%)",
            min_value=0,
            max_value=150,
            value=50,
            help="How much of credit limit is being used?"
        )
    
    with col2:
        quick_late_rate = st.slider(
            "⏰ Late Payment Rate (%)",
            min_value=0,
            max_value=100,
            value=10,
            help="What percentage of payments are late?"
        )
        
        quick_dispute = st.slider(
            "⚠️ Dispute Rate (%)",
            min_value=0,
            max_value=50,
            value=5,
            help="What percentage of invoices have disputes?"
        )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        quick_check = st.button("⚡ QUICK CHECK", use_container_width=True)
    
    if quick_check:
        
        # create customer with defaults
        customer = {
            "avg_monthly_orders": 10.0,
            "total_purchase_amount": 500000.0,
            "avg_order_value": 4166.67,
            "payment_delay_days_avg": float(quick_delay),
            "payment_delay_days_max": float(quick_delay * 1.5),
            "credit_limit": 500000.0,
            "credit_utilization_pct": float(quick_util),
            "num_invoices": 50,
            "num_disputed_invoices": int(50 * quick_dispute / 100),
            "dispute_rate": float(quick_dispute),
            "days_since_first_order": 365,
            "order_frequency_per_month": 5.0,
            "lead_time_variance": 5.0,
            "num_late_payments": int(50 * quick_late_rate / 100),
            "late_payment_rate": float(quick_late_rate),
        }
        
        # predict
        prediction = model.predict_single(customer)
        cat = model.get_risk_category(prediction["default_probability"], mode)
        
        prob_pct = round(prediction["default_probability"] * 100, 1)
        
        # colors
        if cat == "LOW":
            color = "#00ff88"
            emoji = "✅"
            msg = "LOW RISK"
        elif cat == "MEDIUM":
            color = "#ffbb00"
            emoji = "⚠️"
            msg = "MEDIUM RISK"
        elif cat == "HIGH":
            color = "#ff6b35"
            emoji = "🔶"
            msg = "HIGH RISK"
        else:
            color = "#ff0055"
            emoji = "❌"
            msg = "VERY HIGH RISK"
        
        st.markdown("---")
        
        st.markdown(f"""
        <div style='text-align: center; padding: 40px;'>
            <h1 style='font-size: 80px; margin: 0;'>{emoji}</h1>
            <h2 style='color: {color}; font-size: 36px; margin: 10px 0;'>{msg}</h2>
            <p style='color: white; font-size: 24px;'>Default Probability: {prob_pct}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={"suffix": "%", "font": {"size": 50, "color": "white"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "bgcolor": "#162447",
                "steps": [
                    {"range": [0, 30], "color": "rgba(0, 255, 136, 0.3)"},
                    {"range": [30, 50], "color": "rgba(255, 187, 0, 0.3)"},
                    {"range": [50, 70], "color": "rgba(255, 107, 53, 0.3)"},
                    {"range": [70, 100], "color": "rgba(255, 0, 85, 0.3)"}
                ]
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 For detailed analysis with recommendations, use 'New Customer Assessment'")


# risk calculator page
elif page == "📈 Risk Calculator":
    
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1>🧮 Risk Calculator</h1>
        <p style='color: #a0a0a0;'>See how each factor affects credit risk</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Adjust Parameters and See Risk Change")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        
        st.markdown("#### 📊 Risk Factors")
        
        calc_delay = st.slider("Payment Delay (Days)", 0, 90, 20, key="calc1")
        calc_util = st.slider("Credit Utilization (%)", 0, 150, 60, key="calc2")
        calc_late = st.slider("Late Payment Rate (%)", 0, 100, 15, key="calc3")
        calc_dispute = st.slider("Dispute Rate (%)", 0, 50, 5, key="calc4")
        calc_freq = st.slider("Order Frequency (per month)", 1, 30, 8, key="calc5")
    
    with col2:
        
        # calculate risk
        customer = {
            "avg_monthly_orders": float(calc_freq),
            "total_purchase_amount": 500000.0,
            "avg_order_value": 4000.0,
            "payment_delay_days_avg": float(calc_delay),
            "payment_delay_days_max": float(calc_delay * 1.5),
            "credit_limit": 500000.0,
            "credit_utilization_pct": float(calc_util),
            "num_invoices": 50,
            "num_disputed_invoices": int(50 * calc_dispute / 100),
            "dispute_rate": float(calc_dispute),
            "days_since_first_order": 365,
            "order_frequency_per_month": float(calc_freq),
            "lead_time_variance": 5.0,
            "num_late_payments": int(50 * calc_late / 100),
            "late_payment_rate": float(calc_late),
        }
        
        prediction = model.predict_single(customer)
        cat = model.get_risk_category(prediction["default_probability"], mode)
        prob_pct = round(prediction["default_probability"] * 100, 1)
        
        if cat == "LOW":
            color = "#00ff88"
        elif cat == "MEDIUM":
            color = "#ffbb00"
        elif cat == "HIGH":
            color = "#ff6b35"
        else:
            color = "#ff0055"
        
        st.markdown("#### 📈 Live Risk Score")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={"suffix": "%", "font": {"size": 40, "color": "white"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "bgcolor": "#162447",
                "steps": [
                    {"range": [0, 30], "color": "rgba(0, 255, 136, 0.3)"},
                    {"range": [30, 50], "color": "rgba(255, 187, 0, 0.3)"},
                    {"range": [50, 70], "color": "rgba(255, 107, 53, 0.3)"},
                    {"range": [70, 100], "color": "rgba(255, 0, 85, 0.3)"}
                ]
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=300,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        <div style='text-align: center;'>
            <h3 style='color: {color};'>{cat} RISK</h3>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # tips
    st.markdown("### 💡 Risk Reduction Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **To Lower Risk Score:**
        - Reduce payment delays below 15 days
        - Keep credit utilization under 70%
        - Maintain late payment rate under 10%
        - Keep dispute rate under 5%
        """)
    
    with col2:
        st.markdown("""
        **Risk Thresholds:**
        - 🟢 LOW: 0-30% default probability
        - 🟡 MEDIUM: 30-50% default probability
        - 🟠 HIGH: 50-70% default probability
        - 🔴 VERY HIGH: 70%+ default probability
        """)


# how it works page
elif page == "ℹ️ How It Works":
    
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1>ℹ️ How It Works</h1>
        <p style='color: #a0a0a0;'>Understanding the Credit Risk Assessment System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # about
    st.markdown("### 🎯 About This Tool")
    
    st.markdown("""
    This is a **free, universal B2B credit risk assessment tool** that helps businesses 
    evaluate the creditworthiness of their customers.
    
    Whether you're a:
    - 🏭 **Manufacturer** giving credit to distributors
    - 🏪 **Wholesaler** extending credit to retailers
    - 🚛 **Supplier** assessing buyer risk
    - 🏢 **Any B2B business** making credit decisions
    
    This tool provides instant, AI-powered credit assessment.
    """)
    
    st.markdown("---")
    
    # how it works
    st.markdown("### 🔧 Technology Behind")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Machine Learning Model:**
        - Algorithm: XGBoost (Gradient Boosting)
        - Trained on: B2B transaction patterns
        - Accuracy: ~85%
        - Features: 15 business parameters
        """)
    
    with col2:
        st.markdown("""
        **Key Components:**
        - 🤖 **ML Prediction** - Core risk scoring
        - 🔍 **Explainability** - SHAP-based reasoning
        - 🛡️ **Fraud Detection** - Anomaly detection
        - 📊 **Benchmarking** - Similar case comparison
        """)
    
    st.markdown("---")
    
    # factors
    st.markdown("### 📊 Factors We Consider")
    
    factors_data = {
        "Factor": [
            "Payment Delay",
            "Credit Utilization",
            "Late Payment Rate",
            "Dispute Rate",
            "Order Frequency",
            "Relationship Duration",
            "Order Value Patterns",
            "Lead Time Variance"
        ],
        "Weight": ["High", "High", "High", "Medium", "Medium", "Low", "Low", "Low"],
        "Description": [
            "Average days late on payments",
            "How much of credit limit is used",
            "Percentage of late payments",
            "Percentage of disputed invoices",
            "How often customer orders",
            "How long customer relationship",
            "Average and total order values",
            "Consistency in order timing"
        ]
    }
    
    factors_df = pd.DataFrame(factors_data)
    st.table(factors_df)
    
    st.markdown("---")
    
    # risk levels
    st.markdown("### 🎨 Risk Categories")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='background: rgba(0, 255, 136, 0.2); border: 2px solid #00ff88;
                    border-radius: 10px; padding: 20px; text-align: center;'>
            <h3 style='color: #00ff88;'>LOW</h3>
            <p style='color: white;'>0-30%</p>
            <p style='color: #a0a0a0; font-size: 12px;'>Safe to extend credit</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: rgba(255, 187, 0, 0.2); border: 2px solid #ffbb00;
                    border-radius: 10px; padding: 20px; text-align: center;'>
            <h3 style='color: #ffbb00;'>MEDIUM</h3>
            <p style='color: white;'>30-50%</p>
            <p style='color: #a0a0a0; font-size: 12px;'>Approve with caution</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: rgba(255, 107, 53, 0.2); border: 2px solid #ff6b35;
                    border-radius: 10px; padding: 20px; text-align: center;'>
            <h3 style='color: #ff6b35;'>HIGH</h3>
            <p style='color: white;'>50-70%</p>
            <p style='color: #a0a0a0; font-size: 12px;'>Restrict credit limit</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='background: rgba(255, 0, 85, 0.2); border: 2px solid #ff0055;
                    border-radius: 10px; padding: 20px; text-align: center;'>
            <h3 style='color: #ff0055;'>VERY HIGH</h3>
            <p style='color: white;'>70%+</p>
            <p style='color: #a0a0a0; font-size: 12px;'>Decline or cash only</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # disclaimer
    st.markdown("### ⚠️ Disclaimer")
    
    st.warning("""
    This tool provides **indicative risk assessment** based on the data you enter.
    
    - Results are for guidance only, not financial advice
    - Always combine with your own due diligence
    - The model is trained on general B2B patterns
    - Your specific industry may have different risk factors
    
    For critical decisions, consult with financial professionals.
    """)
    
    st.markdown("---")
    
    # contact
    st.markdown("""
    <div style='text-align: center; padding: 30px 0;'>
        <h3 style='color: #00d4ff;'>Built by Aksum</h3>
        <p style='color: #a0a0a0;'>B2B Supply Chain Solutions</p>
        <a href='https://aksum.co.in' target='_blank' 
           style='color: #e94560; font-size: 20px; text-decoration: none;'>
            🌐 aksum.co.in
        </a>
    </div>
    """, unsafe_allow_html=True)


# footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px 0; color: #666;'>
    <p>🏦 Aksum Credit Risk Assessment | Universal B2B Tool</p>
    <p style='font-size: 12px;'>Free to use • No signup required • Instant results</p>
    <p style='font-size: 11px;'>© 2024 <a href='https://aksum.co.in' style='color: #00d4ff;'>aksum.co.in</a></p>
</div>
""", unsafe_allow_html=True)