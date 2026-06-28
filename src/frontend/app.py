import os
import sys
import math
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database.mongo_client import (
    get_targets,
    get_prediction,
    get_model_metrics,
    get_ppi_edges,
    get_overview_stats,
)
from services.uniprot_service import get_entry_info
from services.disgenet_service import get_disease_associations
from services.pubchem_service import resolve_drug_smiles
from services.chemistry_service import (
    is_valid_smiles,
    get_svg,
    compute_properties,
    lipinski_pass,
)
from services.string_service import get_ppi_neighbors

TEMP_DIR = tempfile.gettempdir()

st.set_page_config(
    page_title="DTI Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stApp { background: #f7f8fc; color: #1f2937; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3, h4 { color: #111827; font-family: 'Segoe UI', sans-serif; }
[data-testid="stMetricValue"] {
    color: #2563eb !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #6b7280 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
}
[data-testid="metric-container"] {
    background-color: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 0.95rem;
    font-weight: 500;
    color: #374151 !important;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"]
[data-testid="stMarkdownContainer"] p {
    color: #2563eb !important;
    font-weight: 700;
}
.note-box {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 8px;
    margin-bottom: 10px;
    font-size: 0.85rem;
    color: #374151;
}
footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

def build_recommendation_df(record: dict) -> pd.DataFrame:
    recs = record.get("recommendations", []) if record else []
    rows = []
    for i, rec in enumerate(recs, start=1):
        rows.append(
            {
                "Rank": i,
                "Drug ID / SMILES": rec.get("DrugID", ""),
                "Predicted Score": round(float(rec.get("score", 0)), 4),
                "Chemistry": "Open Chemistry tab",
            }
        )
    return pd.DataFrame(rows)

def build_network_html(target_id: str, recommendations: list) -> str:
    net = Network(
        height="520px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#1f2937",
        directed=False,
    )
    net.repulsion(
        node_distance=180,
        central_gravity=0.15,
        spring_length=200,
        spring_strength=0.05,
        damping=0.9,
    )

    net.add_node(
        target_id,
        label=(target_id[:12] + "…") if len(target_id) > 12 else target_id,
        color={
            "background": "#1e3a8a",
            "border": "#1d4ed8",
            "highlight": {"background": "#2563eb", "border": "#1e3a8a"},
        },
        size=40,
        shape="dot",
        font={"color": "#ffffff", "size": 14, "bold": True},
        title=f"<b>Selected Target</b><br><code>{target_id}</code>",
        physics=False,
        x=0,
        y=0,
    )

    n_drugs = min(len(recommendations), 5)
    for i, rec in enumerate(recommendations[:5]):
        drug = rec.get("DrugID", "?")
        score = float(rec.get("score", 0))
        label = (drug[:10] + "…") if len(drug) > 10 else drug
        angle = (2 * math.pi * i / max(n_drugs, 1)) - math.pi / 2
        net.add_node(
            drug,
            label=label,
            color={
                "background": "#0891b2",
                "border": "#0e7490",
                "highlight": {"background": "#22d3ee", "border": "#0891b2"},
            },
            size=28,
            shape="hexagon",
            font={"color": "#ffffff", "size": 11, "bold": True},
            title=f"<b>Predicted Drug</b><br>ID: <code>{drug}</code><br>Score: <b>{score:.4f}</b>",
            x=round(math.cos(angle) * 280),
            y=round(math.sin(angle) * 280),
        )
        net.add_edge(
            target_id,
            drug,
            value=max(score * 4, 1.0),
            color={"color": "#7dd3fc", "highlight": "#0ea5e9"},
            title=f"ALS predicted score: {score:.4f}",
            width=max(score * 2, 1),
        )

    ppi_edges = get_ppi_edges(target_id)
    if not ppi_edges:
        ppi_edges = get_ppi_neighbors(target_id)
    ppi_edges = ppi_edges[:12]

    ppi_added = set()
    for j, edge in enumerate(ppi_edges):
        a = edge.get("ProteinA")
        b = edge.get("ProteinB")
        conf = float(edge.get("ConfidenceScore", 0))
        other = b if a == target_id else a
        if not other or other in ppi_added:
            continue
        ppi_added.add(other)
        angle = 2 * math.pi * j / max(len(ppi_edges), 1)
        net.add_node(
            other,
            label=other[:12],
            color={
                "background": "#94a3b8",
                "border": "#64748b",
                "highlight": {"background": "#cbd5e1", "border": "#475569"},
            },
            size=18,
            shape="ellipse",
            font={"color": "#1e293b", "size": 10},
            title=(
                f"<b>STRING PPI neighbor</b><br>"
                f"<code>{other}</code><br>"
                f"Confidence: <b>{conf:.3f}</b>"
            ),
            x=round(math.cos(angle) * 480),
            y=round(math.sin(angle) * 480),
        )
        net.add_edge(
            target_id,
            other,
            value=max(conf * 3, 0.5),
            color={"color": "#e2e8f0", "highlight": "#94a3b8"},
            title=f"STRING confidence: {conf:.3f}",
            width=max(conf * 1.5, 0.5),
        )

    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, dir=TEMP_DIR)
    tmp.close()
    net.save_graph(tmp.name)
    raw_html = open(tmp.name, "r", encoding="utf-8").read()
    try:
        os.remove(tmp.name)
    except OSError:
        pass

    return raw_html.replace(
        "</head>",
        """<style>
html,body{margin:0!important;padding:0!important;overflow:hidden!important;background:#ffffff!important;}
#mynetwork{width:100%!important;height:520px!important;border:none!important;background:#ffffff!important;}
</style></head>""",
    )

metrics = get_model_metrics()
stats = get_overview_stats()
all_targets = get_targets()

if not all_targets:
    st.error("No predictions found in MongoDB. Run mongodb_loader.py then als_recommender.py first.")
    st.stop()

with st.sidebar:
    st.header("Target Selection")
    st.divider()

    show_human_only = st.checkbox("🧑 Show human targets only", value=False)
    if show_human_only:
        filtered = [t for t in all_targets if len(t) == 6 and t[0] in ("P", "Q", "O")]
        display_targets = filtered if filtered else all_targets
        st.caption(f"{len(display_targets)} human targets shown")
    else:
        display_targets = all_targets
        st.caption(f"{len(display_targets)} total targets (all organisms)")

    selected_target = st.selectbox("Biological target:", display_targets)
    st.divider()

    st.markdown("**Pipeline**")
    st.markdown("- 6 DTI datasets fused")
    st.markdown("- p-scale: IC50 / Kd / Ki / EC50")
    st.markdown("- Spark ALS (explicit, rank=20)")
    st.markdown("- STRING PPI (confidence ≥ 0.70)")
    st.markdown("- DisGeNET disease context (API)")
    st.divider()

    st.markdown("**DisGeNET setup**")
    st.markdown(
        "1. Register at [disgenet.com](https://www.disgenet.com)\n"
        "2. Copy your API token\n"
        "3. Add to `.env`:\n"
        "```\nDISGENET_API_KEY=your_token\n```"
    )

record = get_prediction(selected_target)
df_table = build_recommendation_df(record)
recs = record.get("recommendations", []) if record else []
disease_df = pd.DataFrame()
gene_sym_used = None

st.title("🧬 Drug–Target Interaction Explorer")
st.caption("Spark ALS recommendation · STRING PPI context · DisGeNET disease enrichment")
st.divider()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Unique Targets", f"{len(all_targets):,}")
m2.metric("Integrated DTI Pairs", f"{stats['n_integrated']:,}")
m3.metric("STRING Edges (≥0.70)", f"{stats['n_ppi']:,}")
m4.metric("Model RMSE", f"{metrics['rmse']:.4f}" if metrics.get("rmse") else "—")
m5.metric("Model AUPR", f"{metrics['aupr']:.4f}" if metrics.get("aupr") else "—")
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Recommendations",
    "🕸 Interaction Network",
    "🔬 Chemistry",
    "🩺 Disease Context",
    "📊 Dataset Analytics",
])

with tab1:
    st.subheader("Top predicted drugs for selected target")
    st.markdown(
        f"<div class='note-box'>Target: <b>{selected_target[:80]}"
        f"{'…' if len(selected_target) > 80 else ''}</b></div>",
        unsafe_allow_html=True,
    )

    if df_table.empty:
        st.warning("No recommendations found for this target.")
    else:
        st.dataframe(
            df_table,
            width="stretch",
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Drug ID / SMILES": st.column_config.TextColumn("Drug ID / SMILES", width="large"),
                "Predicted Score": st.column_config.ProgressColumn(
                    "Predicted Score",
                    format="%.4f",
                    min_value=0.0,
                    max_value=5.0,
                ),
                "Chemistry": st.column_config.TextColumn("Chemistry", width="medium"),
            },
        )

        df_plot = df_table.head(10).copy()
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df_plot["Predicted Score"],
                y=[f"Rank {r}" for r in df_plot["Rank"]],
                orientation="h",
                marker=dict(
                    color=df_plot["Predicted Score"].tolist(),
                    colorscale=[[0, "#93c5fd"], [0.5, "#2563eb"], [1.0, "#1e3a8a"]],
                    cmin=0,
                    cmax=5,
                    showscale=True,
                    colorbar=dict(title="Score", thickness=12),
                ),
                text=df_plot["Predicted Score"].apply(lambda x: f"{x:.4f}"),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Drug: %{customdata}<br>Score: %{x:.4f}<extra></extra>",
                customdata=df_plot["Drug ID / SMILES"].str[:30],
            )
        )
        fig.update_layout(
            height=360,
            margin=dict(l=20, r=80, t=30, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(
                title="ALS Predicted Score",
                range=[0, 5.8],
                gridcolor="#f1f5f9",
            ),
            yaxis=dict(autorange="reversed"),
            font=dict(family="Segoe UI", size=12, color="#1f2937"),
        )
        st.plotly_chart(fig, width="stretch")

        csv = df_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export to CSV",
            data=csv,
            file_name=f"predictions_{selected_target[:12]}.csv",
            mime="text/csv",
        )

        col_v1, col_v2 = st.columns(2, gap="large")

        with col_v1:
            fig_donut = go.Figure(
                go.Pie(
                    labels=["Recommendations", "Other"],
                    values=[len(df_table), 1],
                    hole=0.55,
                    marker=dict(colors=["#16a34a", "#e5e7eb"]),
                    textinfo="label+value",
                )
            )
            fig_donut.update_layout(
                height=260,
                paper_bgcolor="white",
                margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text="Recommendation count", font=dict(size=13)),
                showlegend=False,
            )
            st.plotly_chart(fig_donut, width="stretch")

        with col_v2:
            df_sc = df_table.head(20).copy()
            x = df_sc["Rank"].values
            y = df_sc["Predicted Score"].values

            if len(x) >= 3:
                z = np.polyfit(x, y, deg=2)
                trend_y = np.polyval(z, x)
                fig_sc = go.Figure()
                fig_sc.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=y,
                            colorscale=[[0, "#93c5fd"], [1, "#1e3a8a"]],
                            opacity=0.85,
                            showscale=False,
                        ),
                        name="Score",
                        hovertemplate="Rank %{x} → Score %{y:.4f}<extra></extra>",
                    )
                )
                fig_sc.add_trace(
                    go.Scatter(
                        x=x,
                        y=trend_y,
                        mode="lines",
                        line=dict(color="#dc2626", width=2, dash="dash"),
                        name="Trend",
                    )
                )
            else:
                fig_sc = go.Figure(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=y,
                            colorscale=[[0, "#93c5fd"], [1, "#1e3a8a"]],
                            opacity=0.85,
                            showscale=False,
                        ),
                        name="Score",
                    )
                )

            fig_sc.update_layout(
                height=260,
                paper_bgcolor="white",
                plot_bgcolor="white",
                margin=dict(l=20, r=20, t=30, b=20),
                showlegend=False,
                title=dict(text="Score decay by rank", font=dict(size=13)),
                xaxis=dict(title="Rank", gridcolor="#f1f5f9"),
                yaxis=dict(title="Score", gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig_sc, width="stretch")

with tab2:
    st.subheader("Local interaction network")
    st.markdown(
        "<div class='note-box'>"
        "<span style='color:#1e3a8a;font-weight:700'>●</span> Selected target &nbsp;|&nbsp;"
        "<span style='color:#0891b2;font-weight:700'>⬡</span> Predicted drugs (ALS) &nbsp;|&nbsp;"
        "<span style='color:#94a3b8;font-weight:700'>●</span> STRING PPI neighbors"
        " — edge thickness = interaction strength"
        "</div>",
        unsafe_allow_html=True,
    )

    ppi_in_mongo = get_ppi_edges(selected_target)
    if not ppi_in_mongo:
        st.info("No PPI data in local database for this target — fetching live from STRING DB…")

    with st.spinner("Building network…"):
        html_content = build_network_html(selected_target, recs)
        components.html(html_content, height=540, scrolling=False)

with tab3:
    st.subheader("Lead compound analysis")

    top_smiles = None
    top_drug_id = None
    top_rank = None
    top_drug_name = None

    for _, row in df_table.head(3).iterrows():
        candidate = row["Drug ID / SMILES"]

        if is_valid_smiles(candidate):
            top_smiles = candidate
            top_drug_id = candidate
            top_drug_name = candidate[:30]
            top_rank = int(row["Rank"])
            break

        if candidate.strip().isdigit():
            fetched_smiles, fetched_name = resolve_drug_smiles(candidate)
            if fetched_smiles and is_valid_smiles(fetched_smiles):
                top_smiles = fetched_smiles
                top_drug_id = candidate
                top_drug_name = fetched_name or f"Compound {candidate}"
                top_rank = int(row["Rank"])
                break

    if top_smiles is None:
        st.info(
            "Could not retrieve a valid molecular structure for the top candidates. "
            "The compound IDs may not be valid PubChem CIDs or BindingDB cross-references."
        )
    else:
        st.markdown(
            f"<div class='note-box'>Rank {top_rank} compound: <b>{top_drug_name}</b>"
            f"{f'  (ID: {top_drug_id})' if top_drug_id and top_drug_id.isdigit() else ''}"
            f"</div>",
            unsafe_allow_html=True,
        )
        props = compute_properties(top_smiles)
        svg = get_svg(top_smiles)
        col_mol, col_props = st.columns([1.1, 1.0], gap="large")

        with col_mol:
            st.markdown("**2D Structure**")
            if svg:
                st.markdown(
                    f"<div style='background:white;border:1px solid #e5e7eb;"
                    f"border-radius:10px;padding:12px;text-align:center'>{svg}</div>",
                    unsafe_allow_html=True,
                )
            st.code(top_smiles, language=None)

        with col_props:
            st.markdown("**Molecular properties**")
            if props:
                pc1, pc2 = st.columns(2)
                for idx, (pname, val) in enumerate(props.items()):
                    (pc1 if idx % 2 == 0 else pc2).markdown(
                        f"<div style='background:#f1f5f9;border:1px solid #e2e8f0;"
                        f"border-radius:8px;padding:10px;margin-bottom:8px;'>"
                        f"<div style='font-size:0.75rem;color:#6b7280;text-transform:uppercase;"
                        f"letter-spacing:0.05em'>{pname}</div>"
                        f"<div style='font-size:1.3rem;font-weight:700;color:#1d4ed8'>{val}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                ok = lipinski_pass(props)
                st.markdown(
                    f"<div style='background:{'#dcfce7' if ok else '#fef9c3'};"
                    f"border:1px solid {'#86efac' if ok else '#fde047'};"
                    f"border-radius:8px;padding:12px;margin-top:8px;'>"
                    f"<b style='color:{'#166534' if ok else '#854d0e'}'>"
                    f"{'✓ Lipinski Rule of Five: PASS' if ok else '⚠ Lipinski Rule of Five: CHECK'}"
                    f"</b><br><small style='color:#6b7280'>MW ≤500 · LogP ≤5 · HBD ≤5 · HBA ≤10"
                    f"</small></div>",
                    unsafe_allow_html=True,
                )

                names = list(props.keys())[:5]
                values = [float(props[k]) for k in names]
                fig_r = go.Figure()
                fig_r.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=names,
                        fill="toself",
                        fillcolor="rgba(8,145,178,0.15)",
                        line=dict(color="#0891b2", width=2),
                        name=top_drug_name or "Compound",
                    )
                )
                fig_r.update_layout(
                    polar=dict(radialaxis=dict(visible=True, color="#9ca3af")),
                    showlegend=False,
                    margin=dict(l=30, r=30, t=40, b=20),
                    height=300,
                    paper_bgcolor="white",
                    title=dict(text="Physicochemical profile", font=dict(size=13, color="#374151")),
                )
                st.plotly_chart(fig_r, width="stretch")

with tab4:
    st.subheader("Disease associations — DisGeNET")

    with st.spinner("Resolving UniProt accession…"):
        entry_info = get_entry_info(selected_target)

    badge_color = "#dcfce7" if entry_info["is_human"] else "#fef3c7"
    badge_border = "#86efac" if entry_info["is_human"] else "#fcd34d"
    badge_tc = "#166534" if entry_info["is_human"] else "#92400e"
    badge_label = "✓ Human protein" if entry_info["is_human"] else f"⚠ Non-human: {entry_info['organism']}"

    st.markdown(
        f"<div style='background:{badge_color};border:1px solid {badge_border};"
        f"border-radius:10px;padding:14px 18px;margin-bottom:12px;'>"
        f"<div style='font-size:0.8rem;color:{badge_tc};font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.05em'>{badge_label}</div>"
        f"<div style='margin-top:4px;color:#1f2937'>"
        f"<b>Accession:</b> {selected_target} &nbsp;|&nbsp; "
        f"<b>Protein:</b> {entry_info['protein_name']} &nbsp;|&nbsp; "
        f"<b>Gene:</b> {entry_info['gene_symbol'] or '—'}"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    if not entry_info["is_human"]:
        st.info(
            f"**{selected_target}** is a **{entry_info['organism']}** protein. "
            f"DisGeNET only indexes human gene–disease associations. "
            f"Enable 'Show human targets only' in the sidebar to filter to human proteins."
        )
    else:
        with st.spinner("Querying DisGeNET for disease associations…"):
            disease_df, gene_sym_used = get_disease_associations(selected_target)

        if gene_sym_used:
            st.markdown(
                f"<div class='note-box'>DisGeNET associations for gene "
                f"<b>{gene_sym_used}</b> (mapped from UniProt {selected_target})</div>",
                unsafe_allow_html=True,
            )

        st.dataframe(
            disease_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Disease": st.column_config.TextColumn("Disease", width="large"),
                "Score": st.column_config.NumberColumn("Evidence Score", format="%.4f"),
                "Source": st.column_config.TextColumn("Source", width="medium"),
                "Gene": st.column_config.TextColumn("Gene", width="small"),
            },
        )

        if "Score" in disease_df.columns and disease_df["Score"].notna().any():
            df_dis = (
                disease_df.dropna(subset=["Score"])
                .sort_values("Score", ascending=False)
                .head(12)
            )
            fig_d = px.bar(
                df_dis,
                x="Score",
                y="Disease",
                orientation="h",
                color="Score",
                color_continuous_scale="Blues",
                title=f"Top disease associations for {gene_sym_used or selected_target}",
            )
            fig_d.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                coloraxis_showscale=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_d, width="stretch")

    if entry_info.get("is_human") and "Score" in disease_df.columns and disease_df["Score"].notna().any():
        df_tier = disease_df.dropna(subset=["Score"]).copy()
        df_tier["Tier"] = pd.cut(
            df_tier["Score"],
            bins=[0, 0.2, 0.4, 0.6, 1.0],
            labels=["Low (0–0.2)", "Medium (0.2–0.4)", "High (0.4–0.6)", "Very High (>0.6)"],
        )
        tier_counts = df_tier["Tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]

        fig_tier = px.bar(
            tier_counts,
            x="Tier",
            y="Count",
            color="Tier",
            color_discrete_map={
                "Low (0–0.2)": "#bfdbfe",
                "Medium (0.2–0.4)": "#60a5fa",
                "High (0.4–0.6)": "#2563eb",
                "Very High (>0.6)": "#1e3a8a",
            },
            title="Disease association evidence tiers",
        )
        fig_tier.update_layout(
            height=280,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            xaxis=dict(gridcolor="#f1f5f9"),
            yaxis=dict(gridcolor="#f1f5f9", title="Number of diseases"),
        )
        st.plotly_chart(fig_tier, width="stretch")

with tab5:
    st.subheader("Pipeline & Dataset Overview")

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        source_data = pd.DataFrame(
            {
                "Dataset": [
                    "Davis",
                    "KIBA",
                    "IC50 (BindingDB)",
                    "Kd (BindingDB)",
                    "Ki (BindingDB)",
                    "EC50 (BindingDB)",
                ],
                "Pairs": [30056, 118254, 991000, 321000, 289000, 220000],
            }
        )
        fig_pie = px.pie(
            source_data,
            names="Dataset",
            values="Pairs",
            color_discrete_sequence=[
                "#1e3a8a",
                "#1d4ed8",
                "#2563eb",
                "#3b82f6",
                "#60a5fa",
                "#93c5fd",
            ],
            title="Integrated DTI pairs by source",
            hole=0.42,
        )
        fig_pie.update_traces(textposition="outside", textinfo="label+percent")
        fig_pie.update_layout(
            height=380,
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, width="stretch")

    with col_b:
        raw_nm = np.logspace(-1, 5, 300)
        p_scale = 9 - np.log10(raw_nm)
        fig_ps = go.Figure()
        fig_ps.add_trace(
            go.Scatter(
                x=raw_nm,
                y=p_scale,
                mode="lines",
                line=dict(color="#0891b2", width=2.5),
                name="p-scale",
            )
        )
        fig_ps.add_hline(
            y=6.52,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text="Kd < 30 nM (strong binding)",
            annotation_position="bottom right",
            annotation_font_size=11,
        )
        fig_ps.update_layout(
            title="p-scale: IC50/Kd/Ki/EC50 → affinity score",
            xaxis=dict(title="Raw concentration (nM)", type="log", gridcolor="#f1f5f9"),
            yaxis=dict(title="Affinity score (p-scale)", gridcolor="#f1f5f9"),
            height=380,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_ps, width="stretch")

    st.divider()
    col_c, col_d = st.columns(2, gap="large")

    with col_c:
        pipeline_df = pd.DataFrame(
            {
                "Step": [
                    "1. Load 6 datasets",
                    "2. p-scale transform",
                    "3. Min-Max normalize [1–5]",
                    "4. Union + deduplicate",
                    "5. StringIndexer",
                    "6. ALS matrix factorization",
                    "7. recommendForAllItems",
                    "8. IndexToString decode",
                ],
                "Tool": [
                    "PySpark CSV reader",
                    "log10 transform",
                    "Window function",
                    "groupBy + avg",
                    "MLlib StringIndexer",
                    "MLlib ALS (rank=20)",
                    "ALS.recommendForAllItems(3)",
                    "MLlib IndexToString",
                ],
                "Output": [
                    "6 DataFrames",
                    "p-IC50 / p-Kd / p-Ki / p-EC50",
                    "Score ∈ [1, 5]",
                    "Master DataFrame",
                    "drug_idx / target_idx",
                    "Latent factor matrices U, V",
                    "Top drugs per target",
                    "Real chemical IDs",
                ],
            }
        )
        st.markdown("**Spark ALS Pipeline steps**")
        st.dataframe(pipeline_df, width="stretch", hide_index=True)

    with col_d:
        scores = [float(r.get("score", 0)) for r in recs]
        if scores:
            fig_dist = go.Figure()
            fig_dist.add_trace(
                go.Histogram(
                    x=scores,
                    nbinsx=15,
                    marker_color="#2563eb",
                    opacity=0.85,
                    name="Scores",
                )
            )
            fig_dist.update_layout(
                title=f"Score distribution — {selected_target[:20]}…",
                xaxis=dict(title="Predicted ALS score", gridcolor="#f1f5f9"),
                yaxis=dict(title="Count", gridcolor="#f1f5f9"),
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=340,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_dist, width="stretch")
        else:
            st.info("No score data available for the selected target.")

    st.divider()
    col_e, col_f = st.columns(2, gap="large")

    with col_e:
        quality_data = pd.DataFrame(
            {
                "Dataset": ["Davis", "KIBA", "IC50", "Kd", "Ki", "EC50"],
                "Coverage": [85, 78, 95, 72, 68, 61],
                "Reliability": [92, 88, 75, 89, 84, 71],
                "Scale_Norm": [90, 85, 80, 88, 82, 75],
            }
        )
        fig_heat = go.Figure(
            go.Heatmap(
                z=quality_data[["Coverage", "Reliability", "Scale_Norm"]].values.tolist(),
                x=["Coverage", "Reliability", "Normalizability"],
                y=quality_data["Dataset"].tolist(),
                colorscale=[[0, "#dbeafe"], [0.5, "#3b82f6"], [1, "#1e3a8a"]],
                text=quality_data[["Coverage", "Reliability", "Scale_Norm"]].values.tolist(),
                texttemplate="%{text}%",
                textfont=dict(color="white", size=12),
                showscale=False,
            )
        )
        fig_heat.update_layout(
            title="Dataset quality matrix (estimated scores)",
            height=320,
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=90, b=20),
            xaxis=dict(side="top"),
        )
        st.plotly_chart(fig_heat, width="stretch")

    with col_f:
        datasets = [
            "Davis",
            "KIBA",
            "IC50\n(BindingDB)",
            "Kd\n(BindingDB)",
            "Ki\n(BindingDB)",
            "EC50\n(BindingDB)",
        ]
        pairs = [30056, 118254, 991000, 321000, 289000, 220000]
        cumulative = []
        total = 0
        for p in pairs:
            total += p
            cumulative.append(total)

        fig_cum = go.Figure()
        fig_cum.add_trace(
            go.Scatter(
                x=datasets,
                y=cumulative,
                mode="lines+markers",
                fill="tozeroy",
                fillcolor="rgba(37,99,235,0.10)",
                line=dict(color="#2563eb", width=2.5),
                marker=dict(size=8, color="#1e3a8a"),
            )
        )
        fig_cum.update_layout(
            title="Cumulative DTI pairs across datasets",
            xaxis=dict(title="Dataset (in integration order)", gridcolor="#f1f5f9"),
            yaxis=dict(title="Total pairs", gridcolor="#f1f5f9"),
            height=320,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_cum, width="stretch")