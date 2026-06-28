import streamlit as st
from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME   = os.getenv("MONGO_DB",  "bio_recommender_db")

@st.cache_resource
def get_db():
    return MongoClient(MONGO_URI)[DB_NAME]

def get_targets(limit=500) -> list[str]:
    db   = get_db()
    rows = list(db["predicted_interactions"].find(
        {}, {"TargetID": 1, "_id": 0}
    ).limit(limit))
    return sorted(set(r["TargetID"] for r in rows if "TargetID" in r))

def get_prediction(target_id: str) -> dict | None:
    return get_db()["predicted_interactions"].find_one({"TargetID": target_id})

def get_model_metrics() -> dict:
    return get_db()["model_metrics"].find_one(sort=[("timestamp", -1)]) or {}

def get_ppi_edges(target_id: str, limit=25) -> list:
    return list(get_db()["string_ppi_edges"].find(
        {"$or": [{"ProteinA": target_id}, {"ProteinB": target_id}]},
        {"ProteinA": 1, "ProteinB": 1, "ConfidenceScore": 1, "_id": 0},
    ).limit(limit))

def get_overview_stats() -> dict:
    db = get_db()
    return {
        "n_integrated": db["integrated_interactions"].count_documents({}),
        "n_ppi":        db["string_ppi_edges"].count_documents({}),
    }