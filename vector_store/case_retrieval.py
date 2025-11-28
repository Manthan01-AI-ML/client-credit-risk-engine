# case_retrieval.py
# find similar customers using faiss

import faiss
import numpy as np
import pandas as pd
import os
import sys
import pickle

sys.path.append("..")
import config


class AksumCaseRetrieval:
    
    def __init__(self):
        
        self.index = None
        self.customer_data = None
        self.feature_names = config.FEATURE_NAMES
        self.num_neighbors = config.NUM_NEIGHBORS
        self.vector_dim = config.VECTOR_DIM
        
        print("Aksum Case Retrieval initialized")
    
    
    def build_index(self, customer_df):
        
        print("Building vector index...")
        
        # store original data
        self.customer_data = customer_df.copy()
        
        # get feature columns only
        features = customer_df[self.feature_names]
        
        # convert to numpy array
        vectors = features.values.astype("float32")
        
        # normalize vectors for better similarity
        vectors = self.normalize_vectors(vectors)
        
        # create faiss index
        self.index = faiss.IndexFlatL2(self.vector_dim)
        
        # add vectors to index
        self.index.add(vectors)
        
        num_vectors = self.index.ntotal
        print("Index built with " + str(num_vectors) + " customers")
        
        return self.index
    
    
    def normalize_vectors(self, vectors):
        
        # normalize each feature to 0-1 range
        mins = vectors.min(axis=0)
        maxs = vectors.max(axis=0)
        
        # avoid divide by zero
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        
        normalized = (vectors - mins) / ranges
        
        return normalized.astype("float32")
    
    
    def find_similar(self, customer_data, num_results=5):
        
        # convert dict to dataframe if needed
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
        else:
            df = customer_data
        
        # get features
        features = df[self.feature_names]
        
        # convert to numpy
        query_vector = features.values.astype("float32")
        
        # normalize using stored data stats
        all_features = self.customer_data[self.feature_names].values
        mins = all_features.min(axis=0)
        maxs = all_features.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        
        query_normalized = (query_vector - mins) / ranges
        query_normalized = query_normalized.astype("float32")
        
        # search in index
        distances, indices = self.index.search(query_normalized, num_results)
        
        # get similar customers
        similar_customers = []
        
        for i in range(len(indices[0])):
            
            idx = indices[0][i]
            distance = distances[0][i]
            
            # get customer row
            customer_row = self.customer_data.iloc[idx]
            
            # create result dict
            result = {
                "rank": i + 1,
                "customer_id": customer_row["customer_id"],
                "distance": round(float(distance), 4),
                "similarity_score": round(1 / (1 + float(distance)), 4),
                "default_flag": int(customer_row["default_flag"]),
            }
            
            # add key features
            result["payment_delay_avg"] = customer_row["payment_delay_days_avg"]
            result["credit_utilization"] = customer_row["credit_utilization_pct"]
            result["late_payment_rate"] = customer_row["late_payment_rate"]
            result["dispute_rate"] = customer_row["dispute_rate"]
            
            similar_customers.append(result)
        
        return similar_customers
    
    
    def get_similar_summary(self, similar_customers):
        
        # count defaults
        num_similar = len(similar_customers)
        num_defaulted = 0
        
        for customer in similar_customers:
            if customer["default_flag"] == 1:
                num_defaulted = num_defaulted + 1
        
        num_good = num_similar - num_defaulted
        
        # calculate percentages
        if num_similar > 0:
            default_rate = round((num_defaulted / num_similar) * 100, 1)
        else:
            default_rate = 0
        
        # average similarity
        total_similarity = 0
        for customer in similar_customers:
            total_similarity = total_similarity + customer["similarity_score"]
        
        if num_similar > 0:
            avg_similarity = round(total_similarity / num_similar, 4)
        else:
            avg_similarity = 0
        
        # build summary
        summary = {
            "num_similar_cases": num_similar,
            "num_defaulted": num_defaulted,
            "num_good": num_good,
            "default_rate_pct": default_rate,
            "avg_similarity": avg_similarity,
        }
        
        # add warning if high default rate
        if default_rate >= 50:
            summary["warning"] = "HIGH RISK - Most similar customers defaulted"
        elif default_rate >= 30:
            summary["warning"] = "MEDIUM RISK - Some similar customers defaulted"
        else:
            summary["warning"] = "LOW RISK - Similar customers mostly good"
        
        return summary
    
    
    def print_similar_cases(self, similar_customers, summary):
        
        print("")
        print("=" * 55)
        print("AKSUM - SIMILAR CUSTOMER ANALYSIS")
        print("=" * 55)
        print("")
        
        # print summary first
        print("SUMMARY:")
        print("-" * 40)
        print("Similar cases found: " + str(summary["num_similar_cases"]))
        print("Defaulted: " + str(summary["num_defaulted"]))
        print("Good: " + str(summary["num_good"]))
        print("Default rate: " + str(summary["default_rate_pct"]) + "%")
        print("Avg similarity: " + str(summary["avg_similarity"]))
        print("")
        print("Assessment: " + summary["warning"])
        print("")
        
        # print each similar customer
        print("SIMILAR CUSTOMERS:")
        print("-" * 40)
        
        for customer in similar_customers:
            
            rank = customer["rank"]
            cust_id = customer["customer_id"]
            sim_score = customer["similarity_score"]
            default = customer["default_flag"]
            
            if default == 1:
                status = "DEFAULTED"
            else:
                status = "GOOD"
            
            print("")
            print("Rank " + str(rank) + ": " + cust_id)
            print("  Similarity: " + str(round(sim_score * 100, 1)) + "%")
            print("  Status: " + status)
            print("  Payment Delay Avg: " + str(customer["payment_delay_avg"]) + " days")
            print("  Credit Utilization: " + str(customer["credit_utilization"]) + "%")
            print("  Late Payment Rate: " + str(customer["late_payment_rate"]) + "%")
        
        print("")
        print("=" * 55)
    
    
    def get_text_summary(self, summary, similar_customers):
        
        # build text for llm or display
        parts = []
        
        # intro
        intro = "Analysis of " + str(summary["num_similar_cases"]) + " similar past customers: "
        parts.append(intro)
        
        # default info
        default_info = str(summary["num_defaulted"]) + " defaulted and "
        default_info = default_info + str(summary["num_good"]) + " remained good. "
        parts.append(default_info)
        
        # default rate
        rate_info = "Historical default rate for similar profiles is "
        rate_info = rate_info + str(summary["default_rate_pct"]) + "%. "
        parts.append(rate_info)
        
        # warning
        warning = summary["warning"] + "."
        parts.append(warning)
        
        # top similar customer
        if len(similar_customers) > 0:
            top = similar_customers[0]
            top_info = "Most similar customer (" + top["customer_id"] + ") "
            
            if top["default_flag"] == 1:
                top_info = top_info + "had defaulted with "
            else:
                top_info = top_info + "is in good standing with "
            
            top_info = top_info + str(top["payment_delay_avg"]) + " days avg payment delay."
            parts.append(top_info)
        
        text = " ".join(parts)
        return text
    
    
    def save_index(self, folder_path):
        
        print("Saving vector index...")
        
        # save faiss index
        index_path = os.path.join(folder_path, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        
        # save customer data
        data_path = os.path.join(folder_path, "customer_data.pkl")
        with open(data_path, "wb") as f:
            pickle.dump(self.customer_data, f)
        
        print("Index saved to: " + folder_path)
    
    
    def load_index(self, folder_path):
        
        print("Loading vector index...")
        
        # load faiss index
        index_path = os.path.join(folder_path, "faiss_index.bin")
        self.index = faiss.read_index(index_path)
        
        # load customer data
        data_path = os.path.join(folder_path, "customer_data.pkl")
        with open(data_path, "rb") as f:
            self.customer_data = pickle.load(f)
        
        num_vectors = self.index.ntotal
        print("Index loaded with " + str(num_vectors) + " customers")
        
        return self.index