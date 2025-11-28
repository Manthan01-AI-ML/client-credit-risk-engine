# xgboost_model.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import joblib
import sys

sys.path.append("..")
import config


class AksumCreditModel:
    
    def __init__(self):
        self.model = None
        self.feature_names = config.FEATURE_NAMES
        print("Aksum Credit Model initialized")
    
    
    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        X = df[self.feature_names]
        y = df["default_flag"]
        return X, y
    
    
    def split_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        return X_train, X_test, y_train, y_test
    
    
    def train_model(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        )
        self.model.fit(self.X_train, self.y_train)
        return self.model
    
    
    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        acc = accuracy_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_prob)
        return {"accuracy": round(acc, 4), "auc_roc": round(auc, 4)}
    
    
    def save_model(self, filepath):
        joblib.dump(self.model, filepath)
        print("Model saved")
    
    
    def load_model(self, filepath):
        self.model = joblib.load(filepath)
        print("Model loaded")
    
    
    def predict_single(self, customer_data):
        
        # build feature array in correct order
        features = []
        for name in self.feature_names:
            val = customer_data[name]
            features.append(float(val))
        
        # make 2d numpy array
        X = np.array([features])
        
        # predict
        prob = self.model.predict_proba(X)[0][1]
        pred = self.model.predict(X)[0]
        
        # get category
        cat = self.get_risk_category(prob, "strict")
        
        result = {
            "default_probability": round(float(prob), 4),
            "default_prediction": int(pred),
            "risk_category": cat
        }
        
        return result
    
    
    def get_risk_category(self, prob, mode):
        
        if mode == "strict":
            if prob < 0.3:
                return "LOW"
            elif prob < 0.5:
                return "MEDIUM"
            elif prob < 0.7:
                return "HIGH"
            else:
                return "VERY_HIGH"
        else:
            if prob < 0.4:
                return "LOW"
            elif prob < 0.6:
                return "MEDIUM"
            elif prob < 0.8:
                return "HIGH"
        
            else:
                return "VERY_HIGH"
    
    
    def get_feature_importance(self):
        importance = self.model.feature_importances_
        result = []
        for i in range(len(self.feature_names)):
            result.append((self.feature_names[i], importance[i]))
        result.sort(key=lambda x: x[1], reverse=True)
        return result