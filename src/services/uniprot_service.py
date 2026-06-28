import requests
import streamlit as st

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"

@st.cache_data(ttl=86400)
def get_entry_info(accession: str) -> dict:
    """
    Returns dict with:
      - gene_symbol (str | None)
      - organism    (str)
      - protein_name(str)
      - is_human    (bool)
    """
    result = {
        "gene_symbol":   None,
        "organism":      "Unknown",
        "protein_name":  "Unknown",
        "is_human":      False,
    }
    try:
        resp = requests.get(
            f"{UNIPROT_BASE}/{accession}.json",
            headers={"accept": "application/json"},
            timeout=15,
        )
        if resp.status_code != 200:
            return result
        data = resp.json()

        result["organism"] = (
            data.get("organism", {}).get("scientificName", "Unknown")
        )
        result["is_human"] = (
            data.get("organism", {}).get("taxonId") == 9606
        )
        try:
            result["protein_name"] = (
                data["proteinDescription"]["recommendedName"]["fullName"]["value"]
            )
        except (KeyError, IndexError):
            try:
                result["protein_name"] = (
                    data["proteinDescription"]["submittedNames"][0]["fullName"]["value"]
                )
            except (KeyError, IndexError):
                pass

        genes = data.get("genes", [])
        if genes:
            primary = genes[0].get("geneName", {}).get("value")
            result["gene_symbol"] = primary or (
                genes[0].get("synonyms", [{}])[0].get("value")
            )
    except Exception:
        pass
    return result


# Convenience wrapper kept for backward compatibility
@st.cache_data(ttl=86400)
def to_gene_symbol(accession: str) -> str | None:
    return get_entry_info(accession)["gene_symbol"]