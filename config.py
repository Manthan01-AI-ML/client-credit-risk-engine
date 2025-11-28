# config file for aksum credit risk engine

import os
from pathlib import Path

# load env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# folders
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"
VECTOR_DIR = BASE_DIR / "vector_data"

# make folders
try:
    MODEL_DIR.mkdir(exist_ok=True)
except:
    pass

try:
    VECTOR_DIR.mkdir(exist_ok=True)
except:
    pass

# gemini key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# check key
if GEMINI_API_KEY == "":
    GEMINI_ENABLED = False
elif GEMINI_API_KEY == "paste_your_key_here":
    GEMINI_ENABLED = False
else:
    GEMINI_ENABLED = True

# model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 5
LEARNING_RATE = 0.1

# strict threshold
STRICT_LOW = 0.30
STRICT_MEDIUM = 0.50
STRICT_HIGH = 0.70

# flex threshold
FLEX_LOW = 0.40
FLEX_MEDIUM = 0.60
FLEX_HIGH = 0.80

# features list
FEATURE_NAMES = [
    "avg_monthly_orders",
    "total_purchase_amount",
    "avg_order_value",
    "payment_delay_days_avg",
    "payment_delay_days_max",
    "credit_limit",
    "credit_utilization_pct",
    "num_invoices",
    "num_disputed_invoices",
    "dispute_rate",
    "days_since_first_order",
    "order_frequency_per_month",
    "lead_time_variance",
    "num_late_payments",
    "late_payment_rate",
]

# api settings
API_HOST = "127.0.0.1"
API_PORT = 8000

# vector settings
VECTOR_DIM = 15
NUM_NEIGHBORS = 5

# gemini model
GEMINI_MODEL = "gemini-2.5-flash"

# fraud settings
FRAUD_CONTAMINATION = 0.05

# company
COMPANY_NAME = "Aksum"
CURRENCY = "INR"
MAX_CREDIT = 5000000
MIN_CREDIT = 50000