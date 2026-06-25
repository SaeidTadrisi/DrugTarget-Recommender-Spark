import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network
from pymongo import MongoClient
import pandas as pd

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="DTI Recommender System", layout="wide", page_icon="🧬")

st.title("Drug-Target Interaction (DTI) Predictor")
st.markdown("""
**Big Data Management - Master's Project** This dashboard visualizes novel drug recommendations for target proteins using an **ALS Machine Learning Model** trained on Apache Spark.
""")


# ---------------------------------------------------------
# MongoDB Connection
# ---------------------------------------------------------
@st.cache_resource
def get_database_connection():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["bio_recommender_db"]
    return db


db = get_database_connection()

# ---------------------------------------------------------
# Sidebar & Inputs
# ---------------------------------------------------------
st.sidebar.header("Search Configuration")
st.sidebar.markdown("Select a Target Index to view its AI-predicted drug interactions.")


# Fetch available Target Indices from predictions
@st.cache_data
def get_available_targets():
    collection = db["predicted_interactions"]
    # Limit to 100 for UI performance
    targets = list(collection.find({}, {"target_idx": 1, "_id": 0}).limit(100))
    return sorted([int(t["target_idx"]) for t in targets])


available_targets = get_available_targets()

if not available_targets:
    st.error("No predictions found in MongoDB. Please run the PySpark ALS model first.")
    st.stop()

selected_target = st.sidebar.selectbox("Select Target ID:", available_targets)


# ---------------------------------------------------------
# Fetch Data & Generate Graph
# ---------------------------------------------------------
def generate_interactive_graph(target_idx):
    collection = db["predicted_interactions"]
    record = collection.find_one({"target_idx": target_idx})

    if not record or "recommendations" not in record:
        return None, None

    recommendations = record["recommendations"]

    # Initialize NetworkX Graph
    G = nx.Graph()

    # Add the central Target Node
    target_name = f"Target Protein\n(ID: {target_idx})"
    G.add_node(target_name, size=30, color="#e74c3c", title="Protein Target")  # Red central node

    drug_data = []

    # Add Drug Nodes and Edges
    for rec in recommendations:
        drug_idx = int(rec["drug_idx"])
        score = float(rec["rating"])

        drug_name = f"Drug Candidate\n(ID: {drug_idx})"

        # Add Node: Size and color intensity based on binding score
        node_color = "#3498db" if score > 1.2 else "#95a5a6"  # Blue for high score, Grey for lower
        G.add_node(drug_name, size=20, color=node_color, title=f"Predicted Affinity: {score:.4f}")

        # Add Edge
        G.add_edge(target_name, drug_name, value=score, title=f"Score: {score:.4f}")

        # Save for table
        drug_data.append({"Drug Index": drug_idx, "Predicted Affinity Score": round(score, 4)})

    # Convert to PyVis Network for Interactive HTML
    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(G)

    # Physics settings for biological network look
    net.repulsion(node_distance=150, spring_length=100)

    # Generate HTML file
    path = "frontend/dti_graph.html"
    net.save_graph(path)

    return path, pd.DataFrame(drug_data)


# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
col1, col2 = st.columns([2, 1])

with st.spinner("Generating Biological Network..."):
    graph_path, df_table = generate_interactive_graph(selected_target)

if graph_path:
    with col1:
        st.subheader(f"Interaction Graph for Target {selected_target}")
        st.info(
            "**Tip:** You can drag the nodes, zoom in/out, and hover over links to see the exact prediction scores.")
        # Load the HTML file and display it in Streamlit
        HtmlFile = open(graph_path, 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=550)

    with col2:
        st.subheader("Recommendation Table")
        st.dataframe(df_table.sort_values(by="Predicted Affinity Score", ascending=False), use_container_width=True)
        st.success(
            "These drugs are highly recommended for lab testing (in-vitro) based on the ALS Collaborative Filtering model.")
else:
    st.warning("No recommendations available for this target.")