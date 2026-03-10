import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

ACI_TRAIN = os.path.join(PROJECT_ROOT, "database", "data", "adult.data")
ACI_TEST = os.path.join(PROJECT_ROOT, "database", "data", "adult.test")
BANK_DATA = os.path.join(PROJECT_ROOT, "database", "data_banking", "bank", "bank-full.csv")
CREDIT_DATA = os.path.join(PROJECT_ROOT, "Credit", "credit_data", "crx.data")
