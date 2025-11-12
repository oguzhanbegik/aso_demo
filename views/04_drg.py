# pages_05_drg.py
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

MARKERS_PATH = Path("data/drg_study/TableS2_markers.csv")
PANELS_PATH  = Path("data/drg_study/TableS13_panels.csv")

CLUSTER_NAME_MAP = {
    "Ab-LTMR.NSG2": "Aβ-LTMR (Touch receptor)",
    "Ab-LTMR.LGI2": "Aβ-LTMR (Touch receptor)",
    "Ab-LTMR.ETV1": "Aβ-LTMR (Touch receptor)",
    "Ab-LTMR.CCKAR": "Aβ-LTMR (Touch receptor)",
    "A-LTMR.TAC3": "Aδ-LTMR (Fine touch receptor)",
    "C-LTMR.CDH9": "C-LTMR (C-fiber touch)",
    "A-PEP.NTRK3/S100A16": "Aδ/β-Peptidergic nociceptor",
    "A-PEP.CHRNA7/SLC18A3": "Aβ-Peptidergic cholinergic nociceptor",
    "A-PEP.SCGN/ADRA2C":   "Aβ-Peptidergic cholinergic nociceptor",
    "A-PEP.KIT":           "Aδ/β-Peptidergic nociceptor",
    "C-PEP.TAC1/CACNG5":   "C-Peptidergic nociceptor",
    "C-PEP.TAC1/CHRNA3":   "C-Peptidergic nociceptor",
    "C-PEP.ADORA2B":       "C-Peptidergic nociceptor",
    "C-NP.MRGPRX1/GFRA2":  "C-Nonpeptidergic nociceptor (NP)",
    "C-NP.MRGPRX1/MRGPRX4":"C-Nonpeptidergic nociceptor (NP)",
    "C-NP.SST/CCK":        "C-Nonpeptidergic nociceptor (NP)",
    "C-NP.SST":            "C-Nonpeptidergic nociceptor (NP)",
    "C-Thermo.TRPM8": "C-Thermosensory neuron",
    "C-Thermo.RXFP1": "C-Thermosensory neuron",
    "A-Propr.HAPLN4": "Proprioceptor (muscle spindle)",
    "A-Propr.EPHA3":  "Proprioceptor (muscle spindle)",
    "A-Propr.PCDH8":  "Proprioceptor (muscle spindle)",
    "ATF3": "Injury-activated neuron (regeneration marker)",
}

def _first_present(d: pd.DataFrame, names):
    for n in names:
        if n in d.columns:
            return n
    return None

@st.cache_data(show_spinner=False)
def load_markers(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    col_gene   = _first_present(df, ["Gene","gene","SYMBOL","symbol","GeneSymbol"])
    col_cluster= _first_present(df, ["Cluster","cluster","Cell type","Cell_type","Celltype"])
    col_lfc    = _first_present(df, ["log2FC","log2fc","avg_log2FC","avg_log2fc","logFC"])
    col_padj   = _first_present(df, ["padj","p_adj","p_val_adj","pval_adj","FDR","FDR_p"])
    if not (col_gene and col_cluster and col_lfc and col_padj):
        return pd.DataFrame()
    out = df[[col_gene, col_cluster, col_lfc, col_padj]].copy()
    out.columns = ["Gene","Cluster_raw","log2FC","padj"]
    out["Gene"] = out["Gene"].astype(str).str.strip().str.upper()
    out["Cluster_raw"] = out["Cluster_raw"].astype(str).str.strip()
    out["Cluster"] = out["Cluster_raw"].map(CLUSTER_NAME_MAP).fillna(out["Cluster_raw"])
    out["log2FC"] = pd.to_numeric(out["log2FC"], errors="coerce")
    out["padj"]   = pd.to_numeric(out["padj"], errors="coerce")
    out["neglog10_padj"] = -np.log10(out["padj"].clip(lower=1e-300))
    out = out.dropna(subset=["log2FC","padj"])
    return out

@st.cache_data(show_spinner=False)
def load_panels(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["Gene","Panel"])
    df = pd.read_csv(path)
    col_gene = _first_present(df, ["Gene","gene","SYMBOL","symbol","GeneSymbol"])
    if not col_gene:
        return pd.DataFrame(columns=["Gene","Panel"])
    panel_cols = [c for c in df.columns if c != col_gene]
    long = df.melt(id_vars=[col_gene], value_vars=panel_cols, var_name="Panel", value_name="Present")
    long = long[long["Present"].astype(str).str.lower().isin(["1","true","yes","present"])].copy()
    long["Gene"] = long[col_gene].astype(str).str.upper()
    return long[["Gene","Panel"]].drop_duplicates()

def run(add_to_shortlist_fn):
    st.title("DRG Neuron Cluster Markers — Volcano & Radar")

    markers = load_markers(MARKERS_PATH)
    panels  = load_panels(PANELS_PATH)

    if markers.empty:
        st.error("Failed to load DRG markers table (need Gene, Cluster, log2FC, padj).")
        st.stop()

    all_simplified = sorted(markers["Cluster"].unique().tolist())
    with st.sidebar:
        st.header("Filters")
        sel_clusters = st.multiselect("Simplified clusters", options=all_simplified, default=all_simplified)
        lfc_thr  = st.slider("abs(log2FC) threshold", 0.0, 4.0, 0.5, 0.1)
        padj_thr = st.slider("padj ≤", 1e-6, 0.5, 0.05, 0.000001, format="%.6f")
        volc_cluster = st.selectbox("Volcano cluster", options=sel_clusters if sel_clusters else all_simplified, index=0)

    filt = markers.query("Cluster in @sel_clusters").copy()
    filt = filt[(filt["padj"] <= padj_thr) & (filt["log2FC"].abs() >= lfc_thr)]

    st.subheader("Volcano (single cluster)")
    volc_df = markers[markers["Cluster"] == volc_cluster].copy()
    if volc_df.empty:
        st.info("No rows for the chosen cluster.")
    else:
        volc_df["sig"] = (volc_df["padj"] <= padj_thr) & (volc_df["log2FC"].abs() >= lfc_thr)
        color_seq = np.where(volc_df["sig"], "Significant", "Background")
        fig = px.scatter(
            volc_df, x="log2FC", y="neglog10_padj", color=color_seq,
            hover_data={"Gene": True, "padj": True, "Cluster": True, "neglog10_padj": False},
            color_discrete_map={"Significant":"#d62728","Background":"#7f7f7f"},
            labels={"neglog10_padj":"-log10(padj)"}, height=520
        )
        fig.add_hline(y=-np.log10(padj_thr), line_dash="dot", line_color="#999")
        fig.add_vrect(x0=-lfc_thr, x1=lfc_thr, fillcolor="#bbb", opacity=0.15, line_width=0)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), legend_title_text="", showlegend=True)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Radar — expression bias of a gene across clusters")
    wide = (markers.groupby(["Cluster","Gene"])["log2FC"].mean().reset_index()
                  .pivot(index="Gene", columns="Cluster", values="log2FC")
                  .reindex(columns=all_simplified))
    gene_options = sorted(wide.index.tolist())
    sel_gene = st.selectbox("Select gene", options=gene_options, index=gene_options.index("GFAP") if "GFAP" in gene_options else 0)

    # simple radar via plotly Scatterpolar
    vals = wide.loc[sel_gene, sel_clusters].fillna(0.0) if sel_clusters else wide.loc[sel_gene, all_simplified].fillna(0.0)
    cats = list(vals.index)
    r = list(vals.values)
    r = r + [r[0]]
    cats_closed = cats + [cats[0]]

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=r, theta=cats_closed, fill='toself', name=sel_gene))
    radar.update_layout(polar=dict(radialaxis=dict(visible=True, tickfont=dict(size=10))),
                        showlegend=False, height=520, margin=dict(l=30, r=30, t=30, b=30))
    st.plotly_chart(radar, width="stretch")

    if not panels.empty:
        hits = panels[panels["Gene"] == sel_gene]["Panel"].unique().tolist()
        if hits:
            st.success(f"**{sel_gene}** appears in panels: {', '.join(hits)}")
        else:
            st.caption(f"No panel hits for **{sel_gene}** in Table S13.")

    st.subheader("Filtered marker table")
    st.caption(f"Filters: {len(sel_clusters)} clusters; |log2FC| ≥ {lfc_thr}; padj ≤ {padj_thr}")
    st.dataframe(filt.sort_values(["Cluster","padj"]).reset_index(drop=True), height=380)

    st.download_button(
        "⬇️ Download filtered table (CSV)",
        data=filt.to_csv(index=False).encode(),
        file_name="DRG_markers_filtered.csv",
        mime="text/csv",
        width="stretch"
    )

    # Save for shortlist stats
    st.session_state["drg_markers_table"] = markers.copy()

    # ONLY AT THE END: dropdown to add a gene
    st.divider()
    st.subheader("Add a gene to shortlist")
    add_pick = st.selectbox("Pick one gene", options=sorted(filt["Gene"].unique().tolist()) or sorted(markers["Gene"].unique().tolist()))
    pri = st.slider("Priority", 1, 5, 3, key="drg_pri")
    note = st.text_input("Notes (optional)", key="drg_note")
    if st.button("➕ Add gene", key="drg_add_btn"):
        add_to_shortlist_fn(add_pick, priority=pri, note=note, source="drg_markers")
        st.success(f"Added/updated **{add_pick}**")