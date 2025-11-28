# risk_reasoning.py
# simple version without complex caching

import sys
import os

sys.path.append("..")
import config

try:
    import google.generativeai as genai
    gemini_loaded = True
except:
    gemini_loaded = False
    print("Warning: Google Gemini not loaded")


class AksumLLMAgent:
    
    def __init__(self):
        
        self.model = None
        self.enabled = config.GEMINI_ENABLED
        self.use_gemini = False
        self.api_calls_made = 0
        
        if self.enabled and gemini_loaded:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(config.GEMINI_MODEL)
            print("Aksum LLM Agent ready with Gemini")
        else:
            print("Aksum LLM Agent in fallback mode")
    
    
    def enable_gemini(self):
        if self.enabled and gemini_loaded:
            self.use_gemini = True
            print("Gemini enabled")
    
    
    def disable_gemini(self):
        self.use_gemini = False
        print("Gemini disabled")
    
    
    def get_api_stats(self):
        return {
            "api_calls": self.api_calls_made,
            "cache_size": 0,
            "gemini_enabled": self.use_gemini
        }
    
    
    def generate_decision_explanation(self, customer_data, prediction, explanation, similar_cases):
        
        # always use rule based for now
        parts = []
        
        parts.append("AKSUM CREDIT DECISION")
        parts.append("-" * 30)
        
        risk_cat = prediction["risk_category"]
        risk_score = round(prediction["default_probability"] * 100, 1)
        
        # decision based on category
        if risk_cat == "LOW":
            parts.append("Decision: APPROVED")
            credit_limit = int(customer_data["credit_limit"])
            monitoring = "Quarterly review"
        elif risk_cat == "MEDIUM":
            parts.append("Decision: APPROVED WITH CONDITIONS")
            credit_limit = int(customer_data["credit_limit"] * 0.75)
            monitoring = "Monthly review"
        elif risk_cat == "HIGH":
            parts.append("Decision: RESTRICTED APPROVAL")
            credit_limit = int(customer_data["credit_limit"] * 0.5)
            monitoring = "Weekly monitoring"
        else:
            parts.append("Decision: DECLINED")
            credit_limit = 0
            monitoring = "Cash only"
        
        parts.append("")
        parts.append("Risk Score: " + str(risk_score) + "%")
        parts.append("Credit Limit: INR " + str(credit_limit))
        parts.append("Monitoring: " + monitoring)
        
        text = "\n".join(parts)
        return text