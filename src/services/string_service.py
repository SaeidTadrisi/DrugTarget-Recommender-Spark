import requests
import streamlit as st

STRING_API = "https://string-db.org/api/json"

@st.cache_data(ttl=86400)
def get_ppi_neighbors(uniprot_accession: str, min_score: int = 700) -> list[dict]:
    """
    Fetches PPI neighbors from STRING DB live API.
    Used as fallback when MongoDB has no PPI data for the target.
    Returns list of dicts with keys: ProteinA, ProteinB, ConfidenceScore
    """
    try:
        # First resolve UniProt → STRING ID
        map_resp = requests.post(
            f"{STRING_API}/get_string_ids",
            data={
                "identifiers": uniprot_accession,
                "species":     9606,
                "limit":       1,
                "format":      "json",
            },
            timeout=15,
        )
        if map_resp.status_code != 200:
            return []
        mapping = map_resp.json()
        if not mapping:
            return []
        string_id = mapping[0].get("stringId") or mapping[0].get("preferredName")
        if not string_id:
            return []

        # Fetch interaction partners
        int_resp = requests.post(
            f"{STRING_API}/interaction_partners",
            data={
                "identifiers":    string_id,
                "species":        9606,
                "limit":          25,
                "required_score": min_score,
            },
            timeout=15,
        )
        if int_resp.status_code != 200:
            return []
        interactions = int_resp.json()
        results = []
        for item in interactions:
            results.append({
                "ProteinA":       uniprot_accession,
                "ProteinB":       item.get("preferredName_B", item.get("stringId_B", "")),
                "ConfidenceScore": item.get("score", 0) / 1000,
            })
        return results
    except Exception:
        return []