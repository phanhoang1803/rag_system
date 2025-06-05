import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))


import os
import json
import csv
from typing import List, Dict
from pathlib import Path
from src.ingestion.mongodb_manager import MongoDBManager
from config.settings import settings # Import settings
from pymongo.errors import ConnectionFailure

# Dynamically get the base path for raw data
RAW_DATA_BASE_PATH = settings.BASE_DIR / "data" / "raw"

def load_json_data(filepath: Path) -> List[Dict]:
    """Loads data from a JSON file."""
    if not filepath.exists():
        print(f"Warning: JSON file not found at {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_csv_data(filepath: Path) -> List[Dict]:
    """Loads data from a CSV file."""
    if not filepath.exists():
        print(f"Warning: CSV file not found at {filepath}")
        return []
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def seed_mongodb():
    """Connects to MongoDB and seeds collections with initial data."""
    mongo_manager = MongoDBManager(
        uri=settings.MONGODB_URI,
        db_name=settings.MONGODB_DB_NAME
    )

    try:
        mongo_manager.connect()

        # --- Seed Products Collection ---
        products_data_path = RAW_DATA_BASE_PATH / "structured" / "products.json"
        products_data = load_json_data(products_data_path)
        mongo_manager.insert_documents(settings.MONGODB_PRODUCTS_COLLECTION, products_data, drop_existing=True)

        # --- Seed Employees Collection ---
        employees_data_path = RAW_DATA_BASE_PATH / "structured" / "employees.csv"
        employees_data = load_csv_data(employees_data_path)
        mongo_manager.insert_documents(settings.MONGODB_EMPLOYEES_COLLECTION, employees_data, drop_existing=True)

        # --- Seed FAQs Collection (if desired to live in Mongo instead of documents) ---
        # We'll use a simple hardcoded list for FAQs for this example,
        # but you could load this from a CSV/JSON if you prefer.
        faqs_data = [
            {"question": "How do I request time off?", "answer": "You can request time off through the HR self-service portal under the 'Leave Requests' section. Please submit requests at least 2 weeks in advance for planned leave.", "category": "HR"},
            {"question": "What is our company's mission statement?", "answer": "Our mission is to innovate relentlessly, deliver exceptional value to our customers, and foster a collaborative and inclusive work environment.", "category": "Company Info"},
            {"question": "How do I reset my password?", "answer": "For internal systems, visit the IT self-service portal or contact the IT help desk at extension 123. For external accounts, use the 'Forgot Password' link on the respective service.", "category": "IT Support"},
            {"question": "Where can I find the latest sales figures?", "answer": "The latest sales figures are available on the Sales Dashboard, accessible via the company's internal reporting portal. Access requires sales department credentials.", "category": "Sales"},
            {"question": "What are the company's core values?", "answer": "Our core values include integrity, customer focus, innovation, accountability, and teamwork.", "category": "Company Info"},
            {"question": "What is the process for submitting an expense report?", "answer": "Expense reports are submitted via the finance portal. Please ensure all receipts are attached and categorized correctly. Submissions are due by the 5th of each month.", "category": "Finance"}
        ]
        mongo_manager.insert_documents(settings.MONGODB_FAQS_COLLECTION, faqs_data, drop_existing=True)

    except ConnectionFailure as e:
        print(f"MongoDB seeding failed due to connection error: {e}")
        print("Please ensure MongoDB is running and accessible at the configured URI.")
    except Exception as e:
        print(f"An unexpected error occurred during MongoDB seeding: {e}")
    finally:
        mongo_manager.close()

if __name__ == "__main__":
    # Ensure MongoDB is running before running this script!
    # If using docker-compose, start just MongoDB first: `docker-compose up -d mongodb`
    seed_mongodb()