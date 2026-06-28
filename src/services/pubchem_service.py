import requests
import streamlit as st

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"

@st.cache_data(ttl=86400)
def cid_to_smiles(drug_id: str) -> str | None:
    if not drug_id or not drug_id.strip().isdigit():
        return None
    try:
        resp = requests.get(
            f"{PUBCHEM_BASE}/{drug_id.strip()}/property/IsomericSMILES/JSON",
            timeout=12,
        )
        if resp.status_code == 200:
            props = resp.json().get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("IsomericSMILES")
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400)
def cid_to_name(drug_id: str) -> str | None:
    if not drug_id or not drug_id.strip().isdigit():
        return None
    try:
        resp = requests.get(
            f"{PUBCHEM_BASE}/{drug_id.strip()}/property/IUPACName,Title/JSON",
            timeout=12,
        )
        if resp.status_code == 200:
            props = resp.json().get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("Title") or props[0].get("IUPACName")
    except Exception:
        pass
    return None

@st.cache_data(ttl=86400)
def bindingdb_id_to_smiles(drug_id: str) -> tuple[str | None, str | None]:
    """
    Looks up a BindingDB compound ID via PubChem's BindingDB cross-reference.
    Returns (smiles, name) or (None, None).
    """
    if not drug_id or not drug_id.strip().isdigit():
        return None, None
    try:
        # Try PubChem cross-reference with BindingDB source ID
        resp = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/sourceid/BindingDB/{drug_id.strip()}/property/IsomericSMILES,IUPACName,Title/JSON",
            timeout=15,
        )
        if resp.status_code == 200:
            props = resp.json().get("PropertyTable", {}).get("Properties", [])
            if props:
                smiles = props[0].get("IsomericSMILES")
                name   = props[0].get("Title") or props[0].get("IUPACName")
                return smiles, name
    except Exception:
        pass
    return None, None


@st.cache_data(ttl=86400)
def resolve_drug_smiles(drug_id: str) -> tuple[str | None, str | None]:
    """
    Master resolver: tries PubChem CID first, then BindingDB cross-ref.
    Returns (smiles, name).
    """
    if not drug_id or not drug_id.strip().isdigit():
        # It might already be a SMILES string — caller checks is_valid_smiles
        return None, None

    # Strategy 1: treat as PubChem CID
    smiles = cid_to_smiles(drug_id)
    if smiles:
        name = cid_to_name(drug_id)
        return smiles, name

    # Strategy 2: treat as BindingDB ID via PubChem cross-reference
    smiles, name = bindingdb_id_to_smiles(drug_id)
    if smiles:
        return smiles, name

    return None, None