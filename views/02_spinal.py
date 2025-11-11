# 02_spinal.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from gtex_utils import P, load_gtex_median

def run():
    # everything that builds the page goes here
    # (your existing code in that file)
    ...

SPINAL = "Brain_Spinal_cord_cervical_c-1"

@st.cache_data(show_spinner=False)
def get_gtex():
    return load_gtex_median(P["gtex_median"])  # tissues x genes

def tau(v):
    vmax = v.max()
    if vmax <= 0: return 0.0
    x = 1 - (v / vmax)
    return (x.sum() / (len(v)-1))

def run(add_to_shortlist_fn):
    st.title("Spinal cord specificity — light ranking")

    gtex = get_gtex()
    if gtex.empty:
        st.error("GTEx median TPM not found/parsed (check reference path).")
        st.stop()

    G = gtex.copy()
    if SPINAL not in G.index:
        st.error(f"‘{SPINAL}’ not found in GTEx medians.")
        st.stop()

    spinal = G.loc[SPINAL]
    others = G.drop(index=SPINAL)
    median_other = others.median(axis=0)
    delta = spinal - median_other
    tau_sc = pd.Series({g: tau(G[g].values) for g in G.columns})
    max_other = others.max(axis=0)
    delta = spinal - max_other

    rank = pd.DataFrame({
        "gene": G.columns,
        "spinal_tpm": spinal.values,
        "median_other_tpm": median_other.values,
        "delta_spinal_vs_medianother": delta.values,
        "tau": tau_sc.values
    }).sort_values(["tau","delta_spinal_vs_medianother","spinal_tpm"],
                   ascending=[False, False, False])

    st.caption("High τ + high ΔTPM indicates spinal-enriched candidates.")
    st.dataframe(rank.head(500), use_container_width=True, height=420)

    # ---- Added: density + quick histogram for interpretability
    st.subheader("Distributions (quick sanity checks)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(px.histogram(rank, x="tau", nbins=40, title="τ specificity"), use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(rank, x="delta_spinal_vs_medianother", nbins=40, title="Δ vs max other"), use_container_width=True)
    with c3:
        st.plotly_chart(px.histogram(rank, x="spinal_tpm", nbins=40, title="Spinal TPM"), use_container_width=True)

    # ---- top N table + scatter
    TOPN = 200
    def _top200(df: pd.DataFrame) -> pd.DataFrame:
        order = df.sort_values(
            ["delta_spinal_vs_medianother", "tau", "spinal_tpm", "median_other_tpm"],
            ascending=[False, True, False, True]
        )
        return order.head(TOPN).copy()

    rank_top = _top200(rank)
    st.caption(f"Showing top {len(rank_top)} genes.")
    st.dataframe(rank_top, use_container_width=True, height=420)

    fig = px.scatter(
        rank_top,
        x="spinal_tpm",
        y="median_other_tpm",
        color="tau",
        color_continuous_scale="Viridis",
        hover_data=["gene", "delta_spinal_vs_medianother", "tau"],
        title="Spinal vs median of other tissues (TPM) — top 200 genes"
    )
    fig.add_shape(type="line",
                  x0=0, y0=0,
                  x1=rank_top["spinal_tpm"].max(),
                  y1=rank_top["spinal_tpm"].max(),
                  line=dict(color="gray", dash="dash"))
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title=SPINAL + " (median TPM)",
                      yaxis_title="Median of other tissues (median TPM)")
    st.plotly_chart(fig, use_container_width=True)

    # ---- Save for shortlist page stats
    st.session_state["spinal_rank"] = rank.copy()

    # ---- ONLY AT THE END: dropdown to add to shortlist
    st.divider()
    st.subheader("Add a gene to shortlist")
    pick = st.selectbox("Pick one gene", options=rank_top["gene"].tolist())
    pri = st.slider("Priority", 1, 5, 3)
    note = st.text_input("Notes (optional)")
    if st.button("➕ Add gene"):
        add_to_shortlist_fn(pick, priority=pri, note=note, source="spinal_specificity")
        st.success(f"Added/updated **{pick}**")