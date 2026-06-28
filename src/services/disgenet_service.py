import os
import requests
import pandas as pd
import streamlit as st
from services.uniprot_service import get_entry_info

DISGENET_API   = "https://api.disgenet.com/api/v1"
DISGENET_TOKEN = os.getenv("DISGENET_API_KEY", "")


@st.cache_data(ttl=600)
def get_disease_associations(uniprot_accession: str) -> tuple[pd.DataFrame, str | None]:
    """
    Full pipeline: UniProt accession → organism check → gene symbol → DisGeNET.
    Returns (DataFrame, gene_symbol_or_None).
    """
    entry = get_entry_info(uniprot_accession)

    if not entry["is_human"]:
        return pd.DataFrame([{
            "Disease": (
                f"'{entry['protein_name']}' is a {entry['organism']} protein — "
                f"DisGeNET only covers human gene–disease associations."
            ),
            "Score":  None,
            "Source": "Not applicable",
            "Gene":   "—",
        }]), None

    gene_symbol = entry["gene_symbol"]
    if not gene_symbol:
        return pd.DataFrame([{
            "Disease": f"Human protein {uniprot_accession} has no gene symbol in UniProt.",
            "Score":   None,
            "Source":  "UniProt",
            "Gene":    "—",
        }]), None

    if not DISGENET_TOKEN:
        return pd.DataFrame([{
            "Disease": "⚠ DISGENET_API_KEY not set — see sidebar instructions.",
            "Score":   None,
            "Source":  "—",
            "Gene":    gene_symbol,
        }]), gene_symbol

    headers = {
        "Authorization": f"Bearer {DISGENET_TOKEN}",
        "accept":        "application/json",
    }
    try:
        search_resp = requests.get(
            f"{DISGENET_API}/gene/search",
            params={"gene_symbol": gene_symbol, "source": "ALL"},
            headers=headers,
            timeout=20,
        )
        gda_url = f"{DISGENET_API}/gda/gene/{gene_symbol}"
        if search_resp.status_code == 200:
            gene_data = search_resp.json()
            if gene_data:
                ncbi_id = (
                    gene_data[0].get("ncbi_id")
                    or gene_data[0].get("geneid")
                    or gene_data[0].get("ncbiId")
                )
                if ncbi_id:
                    gda_url = f"{DISGENET_API}/gda/gene/{ncbi_id}"

        gda_resp = requests.get(
            gda_url,
            headers=headers,
            params={"source": "ALL", "format": "json"},
            timeout=20,
        )
        if gda_resp.status_code == 404:
            return pd.DataFrame([{
                "Disease": (
                    f"No disease associations found for gene {gene_symbol} in DisGeNET. "
                    f"This is normal for genes not linked to specific pathologies."
                ),
                "Score":  None,
                "Source": "DisGeNET (no records)",
                "Gene":   gene_symbol,
            }]), gene_symbol

        gda_resp.raise_for_status()
        data = gda_resp.json()

        rows = []
        for item in (data or [])[:25]:
            rows.append({
                "Disease": (
                    item.get("disease_name") or item.get("diseaseName")
                    or item.get("disease") or item.get("diseaseId") or "Unknown"
                ),
                "Score":  item.get("score") or item.get("gda_score"),
                "Source": item.get("source") or item.get("sourceName") or "DisGeNET",
                "Gene":   gene_symbol,
            })
        return pd.DataFrame(rows), gene_symbol

    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        messages = {
            401: "Authentication failed — check your API token.",
            429: "Rate limit reached — wait a moment and retry.",
        }
        msg = messages.get(code, f"HTTP {code} error for gene {gene_symbol}: {e}")
        return pd.DataFrame([{
            "Disease": msg, "Score": None,
            "Source": f"DisGeNET (HTTP {code})", "Gene": gene_symbol,
        }]), gene_symbol
    except Exception as e:
        return pd.DataFrame([{
            "Disease": f"Unexpected error: {e}", "Score": None,
            "Source": "DisGeNET", "Gene": gene_symbol,
        }]), gene_symbol