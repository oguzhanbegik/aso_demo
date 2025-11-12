# pages_03_pain.py
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

HPO_TSV = "data/pain/HP_0012531_associations_export.tsv"
CLINVAR = "data/pain/ShinyClinVar_PAIN_2025-11-07.csv"

@st.cache_data(show_spinner=False)
def load_hpo_pain(tsv_path: str) -> pd.DataFrame:
    if not Path(tsv_path).exists():
        return pd.DataFrame()
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    gene_col = None
    for c in df.columns:
        if c.lower().startswith("mapped"):
            gene_col = c
            break
    if gene_col is None:
        return pd.DataFrame()
    df = df[[gene_col]].dropna()
    df[gene_col] = df[gene_col].astype(str)
    df["GeneSymbol"] = df[gene_col].str.replace(" ", "", regex=False)
    df = df.assign(GeneSymbol=df["GeneSymbol"].str.split("[,|]")).explode("GeneSymbol")
    df["GeneSymbol"] = df["GeneSymbol"].str.upper().str.strip()
    df = df[df["GeneSymbol"] != ""]
    out = (df.groupby("GeneSymbol").size().rename("HPO_pain_hits").reset_index())
    return out

@st.cache_data(show_spinner=False)
def load_clinvar_pain(csv_path: str) -> pd.DataFrame:
    if not Path(csv_path).exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path, dtype=str)
    gcol = None
    for c in df.columns:
        if c.lower() in ("genesymbol","gene_symbol","gene"):
            gcol = c
            break
    if gcol is None:
        return pd.DataFrame()
    df["GeneSymbol"] = df[gcol].astype(str).str.upper().str.strip()
    df = df[df["GeneSymbol"] != ""]
    out = (df.groupby("GeneSymbol").size().rename("ClinVar_pain_variants").reset_index())
    return out

@st.cache_data(show_spinner=False)
def combine_pain_tables(hpo_df, clin_df):
    if hpo_df.empty and clin_df.empty:
        return pd.DataFrame()
    allg = pd.DataFrame({"GeneSymbol": sorted(set(hpo_df["GeneSymbol"]).union(set(clin_df["GeneSymbol"])))})
    out = (allg.merge(hpo_df, on="GeneSymbol", how="left")
                .merge(clin_df, on="GeneSymbol", how="left")).fillna(0)
    out["pain_score"] = out["HPO_pain_hits"].astype(int) + out["ClinVar_pain_variants"].astype(int)
    return out.sort_values(["pain_score","HPO_pain_hits","ClinVar_pain_variants"], ascending=False)

def run(add_to_shortlist_fn):
    st.title("Pain genetics (HPO + ClinVar)")
    st.caption("We summarize **HPO pain-term associations** and **ClinVar pain-labeled variants** by gene. No expression blending here.")

    hpo  = load_hpo_pain(HPO_TSV)
    clin = load_clinvar_pain(CLINVAR)
    pain = combine_pain_tables(hpo, clin)

    if pain.empty:
        st.warning("Could not parse HPO or ClinVar files with gene symbols.")
        st.stop()

    # Quick interpretability blocks
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("HPO genes", f"{len(hpo):,}" if not hpo.empty else "0")
    with c2:
        st.metric("ClinVar genes", f"{len(clin):,}" if not clin.empty else "0")
    with c3:
        st.metric("Union (pain genes)", f"{pain['GeneSymbol'].nunique():,}")

    # Top lists
    st.subheader("Top genes by HPO and ClinVar")
    colA, colB = st.columns(2)
    with colA:
        if not hpo.empty:
            st.plotly_chart(
                px.bar(hpo.sort_values("HPO_pain_hits", ascending=False).head(20), x="HPO_pain_hits", y="GeneSymbol", orientation="h",
                       title="Top HPO pain-associated genes"),
                width="stretch"
            )
    with colB:
        if not clin.empty:
            st.plotly_chart(
                px.bar(clin.sort_values("ClinVar_pain_variants", ascending=False).head(20), x="ClinVar_pain_variants", y="GeneSymbol", orientation="h",
                       title="Top ClinVar pain-variant genes"),
                width="stretch"
            )

    # Combined table
    st.subheader("Combined pain table")
    st.caption("Simple **pain score** = HPO hits + ClinVar variant rows (by gene).")
    st.dataframe(pain.head(500), height=500)

    st.download_button(
        "⬇️ Download pain overlay (CSV)",
        data=pain.to_csv(index=False).encode(),
        file_name="pain_overlay_genes.csv",
        mime="text/csv",
        width="stretch"
    )

    # Save to session for shortlist page stats
    st.session_state["pain_table"] = pain.copy()

    # ONLY AT THE END: dropdown to add a gene
    st.divider()
    st.subheader("Add a gene to shortlist")
    options = pain["GeneSymbol"].tolist()
    pick = st.selectbox("Pick one gene", options=options)
    pri = st.slider("Priority", 1, 5, 3)
    note = st.text_input("Notes (optional)")
    if st.button("➕ Add gene"):
        add_to_shortlist_fn(pick, priority=pri, note=note, source="pain_genetics")
        st.success(f"Added/updated **{pick}**")