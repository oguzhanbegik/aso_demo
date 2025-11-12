# pages_04_shortlist.py
import html
import pandas as pd
import streamlit as st

def _desc(g):
    # Try mygene if online; otherwise blank
    try:
        import mygene
        MG = mygene.MyGeneInfo()
        q = MG.query(g, fields="symbol,name,summary", species="human", size=1)
        if q.get("hits"):
            hit = q["hits"][0]
            txt = hit.get("summary") or hit.get("name") or ""
            if txt:
                # first sentence
                import re
                return re.split(r"(?<=[.!?])\s+", txt.strip())[0]
    except Exception:
        pass
    return ""

def _spinal_stats(gene: str):
    rank = st.session_state.get("spinal_rank")
    if isinstance(rank, pd.DataFrame) and not rank.empty:
        row = rank[rank["gene"]==gene]
        if not row.empty:
            r = row.iloc[0]
            return float(r.get("tau", float("nan"))), float(r.get("delta_spinal_vs_medianother", float("nan"))), float(r.get("spinal_tpm", float("nan")))
    return (float("nan"), float("nan"), float("nan"))

def _pain_stats(gene: str):
    pain = st.session_state.get("pain_table")
    if isinstance(pain, pd.DataFrame) and not pain.empty:
        row = pain[pain["GeneSymbol"]==gene]
        if not row.empty:
            r = row.iloc[0]
            return int(r.get("HPO_pain_hits", 0)), int(r.get("ClinVar_pain_variants", 0)), int(r.get("pain_score", 0))
    return (0, 0, 0)

def _drg_cluster(gene: str):
    drg = st.session_state.get("drg_markers_table")
    if isinstance(drg, pd.DataFrame) and not drg.empty and {"Gene","Cluster","log2FC"}.issubset(drg.columns):
        sub = drg[drg["Gene"]==gene]
        if not sub.empty:
            # dominant cluster by mean log2FC
            grp = sub.groupby("Cluster")["log2FC"].mean().sort_values(ascending=False)
            return grp.index[0]
    return ""

def run(add_to_shortlist_fn, REF_DIR: str):
    st.title("Shortlist")

    sl = st.session_state.get("shortlist", {})
    if not sl:
        st.info("Your shortlist is empty. Add genes from the other pages.")
        st.stop()

    rows = []
    for g, meta in sl.items():
        tau, dlt, tpm = _spinal_stats(g)
        hpo, clv, pain_score = _pain_stats(g)
        drg = _drg_cluster(g)
        rows.append({
            "Gene": g,
            "Priority": meta.get("priority", 3),
            "Description": _desc(g),
            "τ": round(tau, 3) if pd.notna(tau) else "",
            "Δ_vs_max_other": round(dlt, 3) if pd.notna(dlt) else "",
            "SpinalTPM": round(tpm, 3) if pd.notna(tpm) else "",
            "HPO_hits": hpo,
            "ClinVar_hits": clv,
            "Pain_score": pain_score,
            "Top_DRG_cluster": drg,
            "Sources": ", ".join(sorted(meta.get("source", set())))
        })
    df = pd.DataFrame(rows).sort_values(["Priority","Gene"], ascending=[False, True])

    st.subheader("Current shortlist")
    st.dataframe(df, width="stretch", height=440)

    st.divider()
    st.subheader("Edit notes / priority")
    for i, r in df.iterrows():
        c1, c2, c3, c4 = st.columns([2,1,3,1])
        with c1:
            st.markdown(f"**{r['Gene']}**")
        with c2:
            new_p = st.number_input("Priority", 1, 5, int(r["Priority"]), key=f"pri_{r['Gene']}")
        with c3:
            existing = st.session_state["shortlist"][r["Gene"]].get("note","")
            new_n = st.text_input("Notes", existing, key=f"note_{r['Gene']}")
        with c4:
            if st.button("Update", key=f"upd_{r['Gene']}"):
                st.session_state["shortlist"][r["Gene"]]["priority"] = int(new_p)
                st.session_state["shortlist"][r["Gene"]]["note"] = new_n
                st.success(f"Updated {r['Gene']}")
                st.rerun()

    st.divider()
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("⬇️ Download shortlist (CSV)", data=csv_bytes,
                       file_name="aso_shortlist.csv", mime="text/csv")

    # Choose gene to proceed to ASO design
    st.subheader("Proceed to ASO Design")
    proceed_gene = st.selectbox("Pick a gene", options=df["Gene"].tolist())
    if st.button("Open ASO designer"):
        st.session_state["selected_gene"] = proceed_gene
        st.switch_page("app.py")