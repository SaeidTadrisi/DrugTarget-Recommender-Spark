import os
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# ---------------------------------------------------------
# Path Configurations (Dynamic & Robust)
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
GENERATED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "generated")
INPUT_FILE = os.path.join(GENERATED_DATA_DIR, "patient_drug_interactions.csv")

# ---------------------------------------------------------
# MongoDB Configurations
# ---------------------------------------------------------
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "bio_recommender_db"
COLLECTION_NAME = "patients_interactions"


def get_mongo_client():
    """Establish and verify connection to MongoDB."""
    print("Connecting to MongoDB...")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Force a call to the server to verify connection
        client.admin.command('ping')
        print("Successfully connected to MongoDB locally.")
        return client
    except ConnectionFailure:
        print("Error: Could not connect to MongoDB. Is the service running on port 27017?")
        exit(1)


def load_data_to_mongodb():
    """
    ETL Process:
    Extract data from CSV, Transform to Dictionary, Load into MongoDB.
    """
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        print("Please run the data generation script first.")
        exit(1)

    print(f"Reading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # Transform: Convert DataFrame rows to a list of dictionaries (JSON-like format for NoSQL)
    records = df.to_dict(orient='records')
    print(f"Transformed {len(records)} rows into NoSQL documents.")

    client = get_mongo_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    print(f"Loading data into Database: '{DB_NAME}' -> Collection: '{COLLECTION_NAME}'...")

    # Clear the collection first to avoid duplicate data if script is run multiple times
    collection.delete_many({})

    # Load: Insert the data into MongoDB
    collection.insert_many(records)

    print(f"Success! Inserted {len(records)} interaction documents into MongoDB.")

    # Verify insertion
    count = collection.count_documents({})
    print(f"Total documents in collection now: {count}")


def main():
    """Main execution flow for ETL."""
    load_data_to_mongodb()


if __name__ == "__main__":
    main()