# data_generator.py
# creates fake customer data for training

import numpy as np
import pandas as pd
import random

def generate_customer_data(num_customers):
    
    # set seed so we get same data everytime
    np.random.seed(42)
    random.seed(42)
    
    # empty lists to store data
    customer_ids = []
    avg_monthly_orders_list = []
    total_purchase_list = []
    avg_order_value_list = []
    payment_delay_avg_list = []
    payment_delay_max_list = []
    credit_limit_list = []
    credit_util_list = []
    num_invoices_list = []
    num_disputed_list = []
    dispute_rate_list = []
    days_since_first_list = []
    order_freq_list = []
    lead_time_var_list = []
    num_late_list = []
    late_rate_list = []
    default_list = []
    
    # generate data for each customer
    for i in range(num_customers):
        
        # customer id
        cust_id = "CUST" + str(i + 1).zfill(5)
        customer_ids.append(cust_id)
        
        # avg monthly orders - between 1 and 50
        avg_orders = round(random.uniform(1, 50), 1)
        avg_monthly_orders_list.append(avg_orders)
        
        # total purchase amount - between 50000 and 5000000
        total_purchase = round(random.uniform(50000, 5000000), 2)
        total_purchase_list.append(total_purchase)
        
        # avg order value
        if avg_orders > 0:
            avg_value = round(total_purchase / (avg_orders * 12), 2)
        else:
            avg_value = 0
        avg_order_value_list.append(avg_value)
        
        # payment delay avg - between 0 and 60 days
        delay_avg = round(random.uniform(0, 60), 1)
        payment_delay_avg_list.append(delay_avg)
        
        # payment delay max - between delay_avg and 120
        delay_max = round(random.uniform(delay_avg, 120), 0)
        payment_delay_max_list.append(delay_max)
        
        # credit limit - between 100000 and 2000000
        credit_lim = round(random.uniform(100000, 2000000), -3)
        credit_limit_list.append(credit_lim)
        
        # credit utilization - between 0 and 120 percent
        credit_util = round(random.uniform(0, 120), 1)
        credit_util_list.append(credit_util)
        
        # num invoices - between 5 and 200
        num_inv = random.randint(5, 200)
        num_invoices_list.append(num_inv)
        
        # num disputed invoices - between 0 and 20% of invoices
        max_disputed = int(num_inv * 0.2)
        num_disp = random.randint(0, max_disputed)
        num_disputed_list.append(num_disp)
        
        # dispute rate
        if num_inv > 0:
            disp_rate = round((num_disp / num_inv) * 100, 2)
        else:
            disp_rate = 0
        dispute_rate_list.append(disp_rate)
        
        # days since first order - between 30 and 1800
        days_first = random.randint(30, 1800)
        days_since_first_list.append(days_first)
        
        # order frequency per month - between 0.5 and 20
        order_freq = round(random.uniform(0.5, 20), 2)
        order_freq_list.append(order_freq)
        
        # lead time variance - between 0 and 25
        lead_var = round(random.uniform(0, 25), 1)
        lead_time_var_list.append(lead_var)
        
        # num late payments - between 0 and 30
        num_late = random.randint(0, 30)
        num_late_list.append(num_late)
        
        # late payment rate - between 0 and 50 percent
        late_rate = round(random.uniform(0, 50), 2)
        late_rate_list.append(late_rate)
        
        # calculate default flag based on risk factors
        risk_score = 0
        
        # high delay increases risk
        if delay_avg > 30:
            risk_score = risk_score + 2
        elif delay_avg > 15:
            risk_score = risk_score + 1
        
        # high credit utilization increases risk
        if credit_util > 90:
            risk_score = risk_score + 2
        elif credit_util > 70:
            risk_score = risk_score + 1
        
        # high dispute rate increases risk
        if disp_rate > 10:
            risk_score = risk_score + 2
        elif disp_rate > 5:
            risk_score = risk_score + 1
        
        # high late payment rate increases risk
        if late_rate > 30:
            risk_score = risk_score + 2
        elif late_rate > 15:
            risk_score = risk_score + 1
        
        # low order frequency increases risk
        if order_freq < 2:
            risk_score = risk_score + 1
        
        # high lead time variance increases risk
        if lead_var > 15:
            risk_score = risk_score + 1
        
        # decide default based on risk score
        # add some randomness
        random_factor = random.uniform(0, 3)
        final_score = risk_score + random_factor
        
        if final_score > 6:
            default_flag = 1
        else:
            default_flag = 0
        
        default_list.append(default_flag)
    
    # create dataframe
    data = {
        "customer_id": customer_ids,
        "avg_monthly_orders": avg_monthly_orders_list,
        "total_purchase_amount": total_purchase_list,
        "avg_order_value": avg_order_value_list,
        "payment_delay_days_avg": payment_delay_avg_list,
        "payment_delay_days_max": payment_delay_max_list,
        "credit_limit": credit_limit_list,
        "credit_utilization_pct": credit_util_list,
        "num_invoices": num_invoices_list,
        "num_disputed_invoices": num_disputed_list,
        "dispute_rate": dispute_rate_list,
        "days_since_first_order": days_since_first_list,
        "order_frequency_per_month": order_freq_list,
        "lead_time_variance": lead_time_var_list,
        "num_late_payments": num_late_list,
        "late_payment_rate": late_rate_list,
        "default_flag": default_list,
    }
    
    df = pd.DataFrame(data)
    
    return df


def save_data(df, filename):
    
    # save to csv
    df.to_csv(filename, index=False)
    print("Data saved to: " + filename)
    

def load_data(filename):
    
    # load from csv
    df = pd.read_csv(filename)
    print("Data loaded from: " + filename)
    return df


def show_data_info(df):
    
    print("")
    print("=" * 40)
    print("DATA INFORMATION")
    print("=" * 40)
    print("")
    
    # shape
    rows = df.shape[0]
    cols = df.shape[1]
    print("Total rows: " + str(rows))
    print("Total columns: " + str(cols))
    print("")
    
    # columns
    print("Columns:")
    for col in df.columns:
        print("  - " + col)
    print("")
    
    # default distribution
    default_counts = df["default_flag"].value_counts()
    num_good = default_counts.get(0, 0)
    num_bad = default_counts.get(1, 0)
    
    print("Default Distribution:")
    print("  Good customers (0): " + str(num_good))
    print("  Bad customers (1): " + str(num_bad))
    
    bad_percent = round((num_bad / rows) * 100, 2)
    print("  Bad rate: " + str(bad_percent) + "%")
    print("")
    
    # first 5 rows
    print("First 5 rows:")
    print(df.head())
    print("")
    
    print("=" * 40)