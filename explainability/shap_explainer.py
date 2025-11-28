# shap_explainer.py
# explains why model gave that score

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append("..")
import config


class AksumExplainer:
    
    def __init__(self, model):
        
        self.model = model
        self.explainer = None
        self.feature_names = config.FEATURE_NAMES
        
        print("Aksum Explainer created")
    
    
    def setup_explainer(self, background_data):
        
        print("Setting up SHAP explainer...")
        
        # use subset of data for speed on 8gb ram
        if len(background_data) > 100:
            sample_data = background_data.sample(n=100, random_state=42)
        else:
            sample_data = background_data
        
        # create tree explainer for xgboost
        self.explainer = shap.TreeExplainer(self.model)
        
        print("Explainer ready")
        
        return self.explainer
    
    
    def explain_single(self, customer_data):
        
        # customer_data is dictionary
        # convert to dataframe
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
        else:
            df = customer_data
        
        # make sure columns in right order
        df = df[self.feature_names]
        
        # get shap values
        shap_values = self.explainer.shap_values(df)
        
        # get base value
        base_value = self.explainer.expected_value
        
        # create explanation dictionary
        explanation = self.create_explanation(
            shap_values[0], 
            df.iloc[0], 
            base_value
        )
        
        return explanation
    
    
    def create_explanation(self, shap_vals, feature_vals, base_val):
        
        # list to store feature impacts
        impacts = []
        
        for i in range(len(self.feature_names)):
            
            feat_name = self.feature_names[i]
            feat_value = feature_vals[feat_name]
            shap_value = shap_vals[i]
            
            # determine impact direction
            if shap_value > 0:
                direction = "increases_risk"
            elif shap_value < 0:
                direction = "decreases_risk"
            else:
                direction = "no_impact"
            
            impact = {
                "feature": feat_name,
                "value": float(feat_value),
                "shap_value": float(round(shap_value, 4)),
                "direction": direction,
                "abs_impact": float(abs(round(shap_value, 4)))
            }
            
            impacts.append(impact)
        
        # sort by absolute impact
        impacts.sort(key=lambda x: x["abs_impact"], reverse=True)
        
        # get top risk factors
        risk_factors = []
        for item in impacts:
            if item["direction"] == "increases_risk":
                risk_factors.append(item)
        
        # get top positive factors
        positive_factors = []
        for item in impacts:
            if item["direction"] == "decreases_risk":
                positive_factors.append(item)
        
        # build explanation
        explanation = {
            "base_risk_score": float(round(base_val, 4)),
            "all_impacts": impacts,
            "top_3_risk_factors": risk_factors[:3],
            "top_3_positive_factors": positive_factors[:3],
        }
        
        return explanation
    
    
    def print_explanation(self, explanation, customer_id=""):
        
        print("")
        print("=" * 50)
        print("AKSUM CREDIT RISK EXPLANATION")
        if customer_id != "":
            print("Customer: " + str(customer_id))
        print("=" * 50)
        print("")
        
        # base score
        base = explanation["base_risk_score"]
        print("Base Risk Score: " + str(base))
        print("")
        
        # top risk factors
        print("TOP RISK FACTORS (increasing risk):")
        print("-" * 40)
        
        risk_factors = explanation["top_3_risk_factors"]
        
        if len(risk_factors) == 0:
            print("  No major risk factors found")
        else:
            rank = 1
            for factor in risk_factors:
                name = factor["feature"]
                value = factor["value"]
                impact = factor["shap_value"]
                
                print(str(rank) + ". " + name)
                print("   Value: " + str(value))
                print("   Impact: +" + str(impact) + " (increases risk)")
                print("")
                rank = rank + 1
        
        # positive factors
        print("TOP POSITIVE FACTORS (decreasing risk):")
        print("-" * 40)
        
        pos_factors = explanation["top_3_positive_factors"]
        
        if len(pos_factors) == 0:
            print("  No major positive factors found")
        else:
            rank = 1
            for factor in pos_factors:
                name = factor["feature"]
                value = factor["value"]
                impact = factor["shap_value"]
                
                print(str(rank) + ". " + name)
                print("   Value: " + str(value))
                print("   Impact: " + str(impact) + " (decreases risk)")
                print("")
                rank = rank + 1
        
        print("=" * 50)
    
    
    def get_text_explanation(self, explanation, risk_score, risk_category):
        
        # build human readable text
        text_parts = []
        
        # intro
        intro = "This customer has a " + risk_category + " risk rating "
        intro = intro + "with a default probability of " + str(round(risk_score * 100, 1)) + "%."
        text_parts.append(intro)
        
        # risk factors
        risk_factors = explanation["top_3_risk_factors"]
        
        if len(risk_factors) > 0:
            
            risk_text = "The main risk factors are: "
            
            risk_items = []
            for factor in risk_factors:
                name = factor["feature"].replace("_", " ")
                value = factor["value"]
                item = name + " (" + str(value) + ")"
                risk_items.append(item)
            
            risk_text = risk_text + ", ".join(risk_items) + "."
            text_parts.append(risk_text)
        
        # positive factors
        pos_factors = explanation["top_3_positive_factors"]
        
        if len(pos_factors) > 0:
            
            pos_text = "Positive factors include: "
            
            pos_items = []
            for factor in pos_factors:
                name = factor["feature"].replace("_", " ")
                value = factor["value"]
                item = name + " (" + str(value) + ")"
                pos_items.append(item)
            
            pos_text = pos_text + ", ".join(pos_items) + "."
            text_parts.append(pos_text)
        
        # recommendation based on category
        if risk_category == "LOW":
            rec = "Recommendation: Approve credit with standard terms."
        elif risk_category == "MEDIUM":
            rec = "Recommendation: Approve with monitoring and quarterly review."
        elif risk_category == "HIGH":
            rec = "Recommendation: Reduce credit limit and monthly review required."
        else:
            rec = "Recommendation: Decline or require advance payment."
        
        text_parts.append(rec)
        
        # join all parts
        full_text = " ".join(text_parts)
        
        return full_text
    
    
    def save_explanation_plot(self, customer_data, save_path):
        
        # convert to dataframe if needed
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
        else:
            df = customer_data
        
        df = df[self.feature_names]
        
        # get shap values
        shap_values = self.explainer.shap_values(df)
        
        # create waterfall plot
        plt.figure(figsize=(10, 6))
        
        # prepare data for bar chart
        values = shap_values[0]
        names = self.feature_names
        
        # sort by absolute value
        sorted_idx = np.argsort(np.abs(values))[::-1]
        
        top_n = 10
        top_idx = sorted_idx[:top_n]
        
        top_values = [values[i] for i in top_idx]
        top_names = [names[i] for i in top_idx]
        
        # colors based on positive/negative
        colors = []
        for v in top_values:
            if v > 0:
                colors.append("red")
            else:
                colors.append("green")
        
        # create bar chart
        y_pos = range(len(top_names))
        
        plt.barh(y_pos, top_values, color=colors)
        plt.yticks(y_pos, top_names)
        plt.xlabel("SHAP Value (Impact on Risk)")
        plt.title("Aksum Credit Risk - Feature Impact")
        plt.tight_layout()
        
        # save plot
        plt.savefig(save_path)
        plt.close()
        
        print("Explanation plot saved to: " + save_path)
    
    
    def explain_batch(self, customer_df):
        
        # make sure columns in right order
        df = customer_df[self.feature_names]
        
        # get shap values for all
        shap_values = self.explainer.shap_values(df)
        
        # get top feature for each customer
        top_risk_features = []
        top_positive_features = []
        
        for i in range(len(df)):
            
            vals = shap_values[i]
            
            # find highest positive shap (risk)
            max_risk_idx = np.argmax(vals)
            max_risk_feat = self.feature_names[max_risk_idx]
            
            # find lowest negative shap (positive)
            min_risk_idx = np.argmin(vals)
            min_risk_feat = self.feature_names[min_risk_idx]
            
            top_risk_features.append(max_risk_feat)
            top_positive_features.append(min_risk_feat)
        
        # add to dataframe
        result_df = customer_df.copy()
        result_df["top_risk_factor"] = top_risk_features
        result_df["top_positive_factor"] = top_positive_features
        
        return result_df