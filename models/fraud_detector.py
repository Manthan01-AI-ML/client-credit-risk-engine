# fraud_detector.py
# detect fraud patterns in b2b customers

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

sys.path.append("..")
import config


class AksumFraudDetector:
    
    def __init__(self):
        
        self.isolation_forest = None
        self.scaler = None
        self.feature_names = config.FEATURE_NAMES
        self.contamination = config.FRAUD_CONTAMINATION
        self.threshold_scores = {}
        
        print("Aksum Fraud Detector initialized")
    
    
    def train_detector(self, clean_data):
        
        print("Training fraud detector...")
        
        # get features
        X = clean_data[self.feature_names]
        
        # scale the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # train isolation forest
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        
        self.isolation_forest.fit(X_scaled)
        
        # get scores for threshold setting
        scores = self.isolation_forest.score_samples(X_scaled)
        
        # calculate thresholds
        self.threshold_scores["very_suspicious"] = np.percentile(scores, 1)
        self.threshold_scores["suspicious"] = np.percentile(scores, 5)
        self.threshold_scores["unusual"] = np.percentile(scores, 10)
        self.threshold_scores["normal"] = np.percentile(scores, 50)
        
        print("Fraud detector trained")
        print("Thresholds set:")
        for key, value in self.threshold_scores.items():
            print("  " + key + ": " + str(round(value, 4)))
        
        return self.isolation_forest
    
    
    def detect_fraud(self, customer_data):
        
        # convert dict to dataframe if needed
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
        else:
            df = customer_data
        
        # get features
        X = df[self.feature_names]
        
        # scale data
        X_scaled = self.scaler.transform(X)
        
        # get anomaly prediction
        # -1 means anomaly, 1 means normal
        prediction = self.isolation_forest.predict(X_scaled)
        
        # get anomaly score
        # lower score means more anomalous
        score = self.isolation_forest.score_samples(X_scaled)
        
        # determine fraud risk level
        fraud_level = self.get_fraud_level(score[0])
        
        # identify suspicious features
        suspicious_features = self.find_suspicious_features(customer_data)
        
        result = {
            "is_anomaly": int(prediction[0] == -1),
            "anomaly_score": float(round(score[0], 4)),
            "fraud_risk_level": fraud_level,
            "suspicious_features": suspicious_features,
        }
        
        return result
    
    
    def get_fraud_level(self, score):
        
        if score <= self.threshold_scores["very_suspicious"]:
            return "VERY HIGH"
        elif score <= self.threshold_scores["suspicious"]:
            return "HIGH"
        elif score <= self.threshold_scores["unusual"]:
            return "MEDIUM"
        else:
            return "LOW"
    
    
    def find_suspicious_features(self, customer_data):
        
        suspicious = []
        
        # check payment delay
        if customer_data["payment_delay_days_avg"] > 45:
            suspicious.append({
                "feature": "payment_delay_days_avg",
                "value": customer_data["payment_delay_days_avg"],
                "reason": "Very high payment delays"
            })
        
        # check credit utilization
        if customer_data["credit_utilization_pct"] > 100:
            suspicious.append({
                "feature": "credit_utilization_pct",
                "value": customer_data["credit_utilization_pct"],
                "reason": "Over credit limit"
            })
        
        # check dispute rate
        if customer_data["dispute_rate"] > 20:
            suspicious.append({
                "feature": "dispute_rate",
                "value": customer_data["dispute_rate"],
                "reason": "Too many disputes"
            })
        
        # check late payment rate
        if customer_data["late_payment_rate"] > 40:
            suspicious.append({
                "feature": "late_payment_rate",
                "value": customer_data["late_payment_rate"],
                "reason": "Excessive late payments"
            })
        
        # check sudden changes
        if customer_data["lead_time_variance"] > 20:
            suspicious.append({
                "feature": "lead_time_variance",
                "value": customer_data["lead_time_variance"],
                "reason": "Irregular ordering pattern"
            })
        
        # check low order frequency with high credit
        if customer_data["order_frequency_per_month"] < 2 and customer_data["credit_utilization_pct"] > 80:
            suspicious.append({
                "feature": "order_frequency_per_month",
                "value": customer_data["order_frequency_per_month"],
                "reason": "Low orders but high credit usage"
            })
        
        return suspicious
    
    
    def batch_detect(self, customers_df):
        
        # detect fraud for multiple customers
        results = []
        
        for idx, row in customers_df.iterrows():
            customer_dict = row.to_dict()
            result = self.detect_fraud(customer_dict)
            result["customer_id"] = row.get("customer_id", "Unknown")
            results.append(result)
        
        return results
    
    
    def get_fraud_statistics(self, customers_df):
        
        # analyze fraud patterns in portfolio
        
        results = self.batch_detect(customers_df)
        
        total = len(results)
        anomalies = 0
        very_high_risk = 0
        high_risk = 0
        medium_risk = 0
        low_risk = 0
        
        for result in results:
            if result["is_anomaly"] == 1:
                anomalies = anomalies + 1
            
            level = result["fraud_risk_level"]
            if level == "VERY HIGH":
                very_high_risk = very_high_risk + 1
            elif level == "HIGH":
                high_risk = high_risk + 1
            elif level == "MEDIUM":
                medium_risk = medium_risk + 1
            else:
                low_risk = low_risk + 1
        
        stats = {
            "total_customers": total,
            "anomalies_detected": anomalies,
            "anomaly_rate_pct": round(anomalies / total * 100, 2),
            "very_high_risk": very_high_risk,
            "high_risk": high_risk,
            "medium_risk": medium_risk,
            "low_risk": low_risk,
        }
        
        return stats
    
    
    def save_detector(self, folder_path):
        
        print("Saving fraud detector...")
        
        # save isolation forest
        model_path = os.path.join(folder_path, "fraud_detector.pkl")
        joblib.dump(self.isolation_forest, model_path)
        
        # save scaler
        scaler_path = os.path.join(folder_path, "fraud_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        # save thresholds
        threshold_path = os.path.join(folder_path, "fraud_thresholds.pkl")
        joblib.dump(self.threshold_scores, threshold_path)
        
        print("Fraud detector saved to " + folder_path)
    
    
    def load_detector(self, folder_path):
        
        print("Loading fraud detector...")
        
        # load isolation forest
        model_path = os.path.join(folder_path, "fraud_detector.pkl")
        self.isolation_forest = joblib.load(model_path)
        
        # load scaler
        scaler_path = os.path.join(folder_path, "fraud_scaler.pkl")
        self.scaler = joblib.load(scaler_path)
        
        # load thresholds
        threshold_path = os.path.join(folder_path, "fraud_thresholds.pkl")
        self.threshold_scores = joblib.load(threshold_path)
        
        print("Fraud detector loaded from " + folder_path)
        
        return self.isolation_forest
    
    
    def print_fraud_report(self, fraud_result):
        
        print("")
        print("=" * 50)
        print("AKSUM FRAUD DETECTION REPORT")
        print("=" * 50)
        print("")
        
        # anomaly status
        if fraud_result["is_anomaly"] == 1:
            print("STATUS: ANOMALY DETECTED")
        else:
            print("STATUS: Normal Pattern")
        
        print("")
        
        # risk level
        level = fraud_result["fraud_risk_level"]
        print("Fraud Risk Level: " + level)
        
        # color code the risk
        if level == "VERY HIGH":
            print("WARNING: Immediate review required!")
        elif level == "HIGH":
            print("Alert: Manual verification recommended")
        elif level == "MEDIUM":
            print("Note: Monitor this customer closely")
        else:
            print("Assessment: Low fraud risk")
        
        print("")
        
        # anomaly score
        print("Anomaly Score: " + str(fraud_result["anomaly_score"]))
        print("(Lower score = More suspicious)")
        
        print("")
        
        # suspicious features
        suspicious = fraud_result["suspicious_features"]
        
        if len(suspicious) > 0:
            print("Suspicious Patterns Found:")
            print("-" * 40)
            
            for item in suspicious:
                print("")
                print("Feature: " + item["feature"])
                print("  Value: " + str(item["value"]))
                print("  Issue: " + item["reason"])
        else:
            print("No specific suspicious patterns found")
        
        print("")
        print("=" * 50)