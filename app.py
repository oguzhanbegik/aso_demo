# app.py — Plug & Play ASO Decision Demo
import os
import io
import gzip
import re
from typing import List, Tuple, Dict

import pandas as pd
import streamlit as st
from Bio import SeqIO
from Bio.Seq import Seq

FAST_MODE = True          # set False if you want full-length folding
FOLD_LEN_CAP = 4000       # fold at most this many nt (after trimming)
PRESELECT_N = 300         # windows kept before uniqueness check

# --- Lazy fetch the FASTA from HF Hub if missing ---
def ensure_fasta_local(local_path: str) -> str:
    import os
    if os.path.exists(local_path):
        return local_path
    try:
        from huggingface_hub import hf_hub_download
        # create data/reference if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        # adjust repo_id/filename to your dataset repo
        fp = hf_hub_download(
            repo_id="YOUR_USERNAME/GRCh38_reference",  # <- create this dataset once
            filename="Homo_sapiens.GRCh38.dna.primary_assembly.fa",
            repo_type="dataset",
        )
        # symlink/copy into your expected location
        import shutil; shutil.copy(fp, local_path)
        return local_path
    except Exception as e:
        import streamlit as st
        st.error(f"Could not fetch FASTA from Hub: {e}")
        return local_path


# --- Local page modules (make sure these files sit next to app.py) ---
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))  # ensure current dir is on sys.path
# ====== Tiny TOML loader (3.8–3.12) ======
def load_toml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        import tomllib  # py311+
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except Exception:
        try:
            import tomli
            with open(path, "rb") as fh:
                return tomli.load(fh)
        except Exception:
            try:
                import toml
                with open(path, "r", encoding="utf-8") as fh:
                    return toml.load(fh)
            except Exception:
                out, section = {}, None
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("[") and line.endswith("]"):
                            section = line[1:-1].strip()
                            out.setdefault(section, {})
                            continue
                        if "=" in line:
                            k, v = [x.strip() for x in line.split("=", 1)]
                            if v.startswith(("'", '"')) and v.endswith(("'", '"')):
                                v = v[1:-1]
                            if section:
                                out.setdefault(section, {})[k] = v
                            else:
                                out[k] = v
                return out

# ====== FASTA / GTF helpers ======
def load_gtf(path):
    cols = ["seqname","source","feature","start","end","score","strand","frame","attribute"]
    df = pd.read_csv(path, sep="\t", comment="#", names=cols, low_memory=False)
    df = df[df["feature"]=="exon"].copy()

    def attr_get(s, key):
        m = re.search(rf'{key} "([^"]+)"', s)
        return m.group(1) if m else None

    df["gene_name"] = df["attribute"].apply(lambda s: attr_get(s, "gene_name") or attr_get(s, "gene_id"))
    df["transcript_id"] = df["attribute"].apply(lambda s: attr_get(s, "transcript_id"))
    return df[["seqname","start","end","strand","gene_name","transcript_id"]]

@st.cache_data(show_spinner=False)
def load_fasta_dict(fa_path):
    return SeqIO.to_dict(SeqIO.parse(fa_path, "fasta"))

def make_tiles(seq: str, k: int, step: int = 1) -> List[Tuple[int,str,float,str]]:
    tiles = []
    for i in range(0, len(seq)-k+1, step):
        tile = seq[i:i+k].upper()
        gc = (tile.count("G")+tile.count("C"))/k
        flags = []
        if "GGGG" in tile: flags.append("G4")
        if "TTTT" in tile: flags.append("T4")
        if "CG" in tile:   flags.append("CpG")
        tiles.append((i, tile, round(gc,3), ";".join(flags)))
    return tiles

# ====== Config ======
CONFIG_PATH = "config.toml"
cfg = load_toml(CONFIG_PATH)

DATA_DIR      = cfg.get("paths", {}).get("data_dir", "data")
PAIN_DIR      = os.path.join(DATA_DIR, "pain")
DRG_STUDY_DIR = os.path.join(DATA_DIR, "drg_study")
REF_DIR       = os.path.join(DATA_DIR, "reference")

# ====== Session state (single, unified) ======
# Shortlist schema: { "GENE": {"priority": int|None, "note": str, "source": set[str]} }
if "shortlist" not in st.session_state:
    st.session_state.shortlist = {}
if isinstance(st.session_state.shortlist, set):
    st.session_state.shortlist = {g: {"priority": None, "note": "", "source": set()} for g in st.session_state.shortlist}

def shortlist_preview_text(max_chars: int = 120) -> str:
    genes = sorted(st.session_state.shortlist.keys())
    text = ", ".join(genes) if genes else "—"
    return text if len(text) <= max_chars else text[:max_chars] + "…"

def add_to_shortlist(genes, *, priority=None, note="", source=None):
    if isinstance(genes, str):
        genes = [genes]
    tag = source if isinstance(source, str) else None
    for g in [str(x).strip().upper() for x in genes if str(x).strip()]:
        entry = st.session_state.shortlist.get(g, {"priority": None, "note": "", "source": set()})
        if priority is not None:
            entry["priority"] = int(priority)
        if note:
            entry["note"] = note
        if tag:
            entry.setdefault("source", set()).add(tag)
        else:
            entry.setdefault("source", set())
        st.session_state.shortlist[g] = entry

def remove_from_shortlist(genes):
    if isinstance(genes, str):
        genes = [genes]
    for g in genes:
        st.session_state.shortlist.pop(str(g).strip().upper(), None)

# ====== Generic readers ======
@st.cache_data(show_spinner=False)
def read_tsv(path, **kwargs):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as fh:
            return pd.read_csv(fh, sep="\t", **kwargs)
    return pd.read_csv(path, sep="\t", **kwargs)

@st.cache_data(show_spinner=False)
def read_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs)

@st.cache_data(show_spinner=False)
def read_excel(path, **kwargs):
    return pd.read_excel(path, **kwargs)

def gene_col(df):
    candidates = ["gene","Gene","symbol","Symbol","SYMBOL","GeneSymbol","GENE","hgnc_symbol","ensembl_gene_id","Ensembl","ENSG"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            return c
    return df.columns[0]

# ====== Page 1: HOME ======
def page_home():
    st.title("Plug-and-Play ASO Decision Demo (Pain/DRG)")
    st.markdown("""
This app helps you **prioritize antisense oligonucleotide (ASO) targets** for **pain** using public and study-specific evidence.  
The workflow is split across pages; **use the sidebar** to navigate.

### Pipeline
1. **SpinalCord Specificity**  
   Screen for genes enriched in spinal cord (and DRG proxies) using GTEx median TPM. You’ll see an interpretable ranking (τ specificity, ΔTPM).
2. **Pain Genetics**  
   Overlay **HPO** and **ClinVar** pain evidence. Quick summaries and counts make it obvious which genes recur across sources.
3. **DRG Cluster Markers**  
   Validate the candidates against **DRG single-cell** markers (volcano/radar), and see if they appear in published panels.
4. **Shortlist**  
   Central board to review all genes you added. You’ll see per-gene quick stats (specificity, pain hits, dominant DRG cluster) and a short description.
5. **ASO Design**  
   For a chosen gene, preview transcript tiling, isoform-level TPMs, a **fast isoform schematic**, **secondary structure (arc & circle)**, and **ready-to-order ASO tables**.

**How to use**  
- On each page, **scroll to the end** to add a gene via a dropdown to the shared **Shortlist**.  
- In **Shortlist**, pick a gene to jump into **ASO Design**.  
- Use `config.toml` to point to your local files (FASTA/GTF, GTEx, DRG tables, etc.).

**Paths (from config):**  
- Data: `{DATA_DIR}`  
- Reference: `{REF_DIR}` (expects `*.fa`, `*.gtf`, GTEx `.gct.gz`)  
- Pain: `{PAIN_DIR}`  
- DRG study: `{DRG_STUDY_DIR}`
""")

# ====== Page 2 loader (for shortlist stats later) ======
def load_spinalcord_specificity():
    gtex_path = os.path.join(REF_DIR, "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_median_tpm.gct.gz")
    df = read_tsv(gtex_path, comment="#", low_memory=False)
    if "Description" in df.columns:
        df.rename(columns={"Description": "SYMBOL"}, inplace=True)
    if "Name" in df.columns:
        df.rename(columns={"Name": "ENSG"}, inplace=True)
    keep = ["ENSG", "SYMBOL", "Spinal_cord_cervical_c_1"]
    cols = [c for c in keep if c in df.columns]
    if cols:
        df = df[cols].copy()
        df.rename(columns={"Spinal_cord_cervical_c_1": "SpinalCord_TPM"}, inplace=True)
    return df

# ====== Page 3 loader ======
def load_pain_genetics():
    hpo = os.path.join(PAIN_DIR, "HP_0012531_associations_export.tsv")
    clinvar = os.path.join(PAIN_DIR, "ShinyClinVar_Pain_2025-11-07.csv")
    dfs = []
    if os.path.exists(hpo):
        d1 = read_tsv(hpo, low_memory=False)
        dfs.append(d1)
    if os.path.exists(clinvar):
        d2 = read_csv(clinvar, low_memory=False)
        dfs.append(d2)
    if not dfs:
        return pd.DataFrame(columns=["SYMBOL"])
    for i, d in enumerate(dfs):
        gc = gene_col(d)
        dfs[i] = d.rename(columns={gc: "SYMBOL"})
        dfs[i]["SOURCE"] = os.path.basename([hpo, clinvar][i])
    out = pd.concat(dfs, ignore_index=True).dropna(subset=["SYMBOL"]).drop_duplicates()
    return out

# ====== Page 4 loader ======
def load_drg_markers():
    t2 = os.path.join(DRG_STUDY_DIR, "TableS2_markers.csv")
    t13 = os.path.join(DRG_STUDY_DIR, "TableS13_panels.csv")
    dfs = []
    if os.path.exists(t2):
        dfs.append(read_csv(t2, low_memory=False))
    if os.path.exists(t13):
        dfs.append(read_csv(t13, low_memory=False))
    if not dfs:
        return pd.DataFrame(columns=["SYMBOL"])
    for i, d in enumerate(dfs):
        dfs[i] = d.rename(columns={gene_col(d): "SYMBOL"})
        dfs[i]["SOURCE"] = os.path.basename([t2, t13][i if i < 2 else 1])
    out = pd.concat(dfs, ignore_index=True).dropna(subset=["SYMBOL"]).drop_duplicates()
    return out


def _normalize_chrom(chrom: str, ref_dict):
    """Ensure chromosome names match between GFF ('chr17') and FASTA ('17') or vice versa."""
    if chrom in ref_dict:
        return chrom
    # try removing or adding 'chr'
    alt = chrom.replace("chr", "") if chrom.startswith("chr") else f"chr{chrom}"
    if alt in ref_dict:
        return alt
    # last resort: try stripping version suffix
    alt2 = chrom.split(".")[0]
    if alt2 in ref_dict:
        return alt2
    return chrom  # fallback, will error if really missing



# ==========================
# ASO DESIGN (GFF-only, fast)
# ==========================

# --- speed knobs ---
FAST_MODE     = True        # fold a trimmed cDNA to keep it snappy
FOLD_LEN_CAP  = 4000        # max nt to fold in FAST_MODE
PRESELECT_N   = 300         # windows kept before uniqueness check
TOP_N_ASOS    = 10          # final ASO count


def arc_plot_interactive(struct: str, highlights: list[tuple[int,int]], title="Secondary structure — arc (interactive)"):
    import numpy as np, plotly.graph_objects as go
    n = len(struct)
    pairs = dotbracket_pairs(struct)

    fig = go.Figure()
    # baseline
    fig.add_trace(go.Scatter(x=[0, n-1], y=[0, 0], mode="lines",
                             line=dict(color="#999", width=1), hoverinfo="skip"))
    # arcs
    for i, j in pairs:
        r = (j - i) / 2.0
        if r <= 0:
            continue
        xm = (i + j) / 2.0
        t  = np.linspace(0.0, 1.0, 60)
        x  = i + (j - i) * t
        y  = np.sqrt(np.clip(1 - ((x - xm)/r)**2, 0, None))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                 line=dict(color="#666", width=1), hoverinfo="skip"))
    # highlight ASO spans
    for s, e in highlights:
        fig.add_shape(type="rect", x0=s, x1=e, y0=-0.04, y1=0.04,
                      line=dict(width=0), fillcolor="#1f77b4", opacity=0.45)

    fig.update_layout(
        title=title, showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(range=[-1, n], showgrid=False, zeroline=False),
        yaxis=dict(range=[-0.05, 1.05], showgrid=False, zeroline=False),
        dragmode="pan"
    )
    return fig


def circle_plot_interactive(struct: str, highlights: list[tuple[int,int]], title="Secondary structure — circle (interactive)"):
    import numpy as np, plotly.graph_objects as go
    n = len(struct)
    pairs = dotbracket_pairs(struct)

    theta = 2 * np.pi * (np.arange(n) / n)
    x, y = np.cos(theta), np.sin(theta)

    fig = go.Figure()
    # nodes
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
                             marker=dict(size=3, color="#777"), hoverinfo="skip"))
    # chords (base pairs)
    for i, j in pairs:
        t = np.linspace(0, 1, 50)
        xi = x[i] * (1 - t) + x[j] * t
        yi = y[i] * (1 - t) + y[j] * t
        fig.add_trace(go.Scatter(x=xi, y=yi, mode="lines",
                                 line=dict(color="#888", width=1), hoverinfo="skip"))
    # highlighted ASO chords
    for s, e in highlights:
        t = np.linspace(0, 1, 80)
        xi = x[s] * (1 - t) + x[e] * t
        yi = y[s] * (1 - t) + y[e] * t
        fig.add_trace(go.Scatter(x=xi, y=yi, mode="lines",
                                 line=dict(color="#1f77b4", width=3), hoverinfo="skip"))

    fig.update_layout(
        title=title, showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(scaleanchor="y", showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        dragmode="pan"
    )
    return fig

# ---------- structure utils ----------
def dotbracket_pairs(struct: str):
    stack, pairs = [], []
    for i, ch in enumerate(struct):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            j = stack.pop()
            pairs.append((j, i))
    return pairs

def compute_pairing_depth(struct: str):
    """Per-position arc depth from dot-bracket structure (O(n))."""
    import numpy as np
    depth = np.zeros(len(struct), dtype=np.int32)
    stack = []
    for i, ch in enumerate(struct):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                j = stack.pop()
                depth[j:i+1] += 1
    return depth

def _rank_windows_fast(seq_U: str, struct: str, win: int, preselect: int = 300):
    """
    Vectorized window ranking by open fraction and low arc depth.
    Returns top `preselect` windows for later uniqueness/Tm/GC filtering.
    """
    import numpy as np
    n = len(struct)
    if n < win:
        return pd.DataFrame()

    # '.' → 1.0
    is_open = (np.frombuffer(struct.encode(), dtype='S1') == b'.').astype(np.float32)
    depth   = compute_pairing_depth(struct).astype(np.float32)

    k = np.ones(win, dtype=np.float32)
    open_sum  = np.convolve(is_open, k, mode='valid')
    depth_sum = np.convolve(depth,   k, mode='valid')

    open_frac  = open_sum / win
    mean_depth = depth_sum / win

    # invert depth to [0..1], higher is better (less paired)
    if mean_depth.max() > mean_depth.min():
        depth_penalty = 1.0 - (mean_depth - mean_depth.min()) / (mean_depth.max() - mean_depth.min())
    else:
        depth_penalty = np.ones_like(mean_depth)

    score_struct = open_frac * depth_penalty

    # preselect
    order = np.argsort(-score_struct)[:max(preselect, 10)]
    rows = []
    for idx in order:
        s, e = int(idx), int(idx + win)
        rna = seq_U[s:e]
        dna = rna.replace("U", "T")
        # filter long homopolymers (>=5)
        if any(b*5 in dna for b in "ATGC"):
            continue
        rows.append({
            "start": s, "end": e, "sequence_RNA": rna,
            "open_frac": float(open_frac[idx]),
            "mean_depth": float(mean_depth[idx]),
            "score_struct": float(score_struct[idx]),
        })
    return pd.DataFrame(rows)

def antisense_from_dna(target_dna: str) -> str:
    # target is RNA window; we replaced U→T already when calling this
    return str(Seq(target_dna).reverse_complement())

# --- gray structure plots with highlighted ASO spans ---
def arc_plot_gray(struct: str, highlights: list[tuple[int,int]], title="Secondary structure — arc"):
    import matplotlib.pyplot as plt
    pairs = dotbracket_pairs(struct)
    fig, ax = plt.subplots(figsize=(10, 3))
    # baseline
    ax.hlines(0, 0, len(struct)-1, color="#999", linewidth=1)
    # arcs
    for (i, j) in pairs:
        xm = (i + j) / 2.0
        r  = (j - i) / 2.0
        if r <= 0: 
            continue
        t  = pd.Series([x/60.0 for x in range(61)]).values
        x  = i + (j - i)*t
        y  = (1 - ((x - xm)/r)**2)**0.5
        ax.plot(x, y, linewidth=0.8, color="#666")
    # highlight ASO spans (simple thick baseline segments)
    for (s, e) in highlights:
        ax.plot([s, e], [0, 0], linewidth=6, solid_capstyle="butt", color="#1f77b4")
    ax.set_xlim(-1, len(struct))
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    ax.set_title(title)
    return fig

def circle_plot_gray(struct: str, highlights: list[tuple[int,int]], title="Secondary structure — circle"):
    import matplotlib.pyplot as plt, numpy as np
    pairs = dotbracket_pairs(struct)
    n = len(struct)
    theta = 2*np.pi*(np.arange(n)/n)
    x, y = np.cos(theta), np.sin(theta)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(x, y, s=4, color="#777")
    for (i, j) in pairs:
        t = np.linspace(0, 1, 50)
        ax.plot(x[i]*(1-t)+x[j]*t, y[i]*(1-t)+y[j]*t, linewidth=0.6, color="#888", alpha=0.9)
    # overlay highlights as thick chords
    for (s, e) in highlights:
        t = np.linspace(0, 1, 80)
        ax.plot(x[s]*(1-t)+x[e]*t, y[s]*(1-t)+y[e]*t, linewidth=3, color="#1f77b4")
    ax.axis("equal"); ax.axis("off"); ax.set_title(title)
    return fig

def plot_pairing_depth_line(struct: str, top_windows: pd.DataFrame, title="Pairing depth (lower is better)"):
    import numpy as np, plotly.graph_objects as go
    depth = compute_pairing_depth(struct).astype(float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(depth)), y=depth, mode="lines", name="pairing depth",
        line=dict(color="gray")
    ))
    # interactive labels for ASO windows
    for i, r in top_windows.iterrows():
        s, e = int(r["start"]), int(r["end"])
        fig.add_vrect(x0=s, x1=e, fillcolor="royalblue", opacity=0.25, line_width=0)
        fig.add_annotation(
            x=(s+e)//2, y=max(depth[s:e]) if e>s else 0,
            text=f"{i+1}. {s}-{e}<br>{r['ASO_DNA_5to3']}",
            showarrow=False, yshift=18, font=dict(size=10), bgcolor="white"
        )
    fig.update_layout(
        title=title, xaxis_title="nt (stitched cDNA)", yaxis_title="arc depth",
        margin=dict(l=20,r=20,t=40,b=10), height=320
    )
    return fig

def page_aso_design():
    from gtex_utils import (
        load_sup1a, parse_gff_fast, load_genome_dict_normalized,
        rnafold_structure, uniqueness_vs_isoforms, wallace_tm,
        fetch_gene_isoforms, gene_isoform_summary, plot_replicate_tpms_matplotlib,
        plot_isoform_schematic_fast  # we keep the fast schematic (GFF-based)
    )

    st.header("ASO Design (GFF-only)")

    # Paths
    #fa        = os.path.join(REF_DIR, "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    fa = os.path.join(REF_DIR, "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    fa = ensure_fasta_local(fa)
    gff       = os.path.join(DATA_DIR, "raw", "LRS_hDRG_clustered.aligned.collapsed.gff")
    sup1a_xls = os.path.join(DATA_DIR, "raw", "SupplementaryTable_1A.xlsx")

    if not (os.path.exists(fa) and os.path.exists(gff)):
        st.warning("Need reference FASTA and GFF. Check config/data paths.")
        return

    # Gene selector
    genes = sorted(st.session_state.shortlist.keys())
    default_gene = st.session_state.get("selected_gene") or (genes[0] if genes else "")
    c1, c2 = st.columns([2,1])
    with c1:
        gene = st.selectbox("Gene", options=(genes or [""]), index=genes.index(default_gene) if default_gene in genes else 0)
    with c2:
        k = st.slider("ASO length", 15, 22, 18, 1)
    if not gene:
        st.info("Shortlist a gene first, then select it here.")
        return

    # Load genome + isoforms from GFF
    ref = load_genome_dict_normalized(fa)
    tx_map = parse_gff_fast(gff)
    sup1a  = load_sup1a(sup1a_xls) if os.path.exists(sup1a_xls) else pd.DataFrame()
    isoforms = fetch_gene_isoforms(tx_map, sup1a, gene)  # list of (pbid, exons, chr, strand)

    # ---- Isoform TPMs (if available)
    st.subheader("Isoform-level expression (TPM)")
    if not sup1a.empty:
        table = gene_isoform_summary(sup1a, gene)
        if not table.empty:
            st.dataframe(table, use_container_width=True, height=300)
            fig = plot_replicate_tpms_matplotlib(table, gene)
            if fig: st.pyplot(fig, clear_figure=True, use_container_width=True)
    else:
        st.caption("Sup1A not found; isoform TPMs skipped.")

    # ---- Isoform schematic (GFF-only)
    st.subheader("Isoform schematic")
    if isoforms:
        target_pbid = None
        # choose target: highest TPM_mean if available else longest
        try:
            if not sup1a.empty:
                sc = gene_isoform_summary(sup1a, gene)
                if not sc.empty:
                    target_pbid = sc.iloc[0]["pbid"]
        except Exception:
            pass
        if not target_pbid:
            # pick longest by total exon span
            spans = [(pb, sum(e2-e1 for (e1,e2) in ex), ch, stn) for (pb, ex, ch, stn) in isoforms]
            spans.sort(key=lambda x: x[1], reverse=True)
            target_pbid = spans[0][0]

        fig = plot_isoform_schematic_fast(isoforms, target_pbid=target_pbid, gtf_regions=None, title=f"{gene} — isoforms (GFF)")
        if fig: st.pyplot(fig, clear_figure=True, use_container_width=True)
    else:
        st.warning("No isoforms found for this gene in GFF."); return

    # ---- Build cDNA of the target isoform only (GFF → exons)
    t_rec = next((t for t in isoforms if t[0].split("|")[0] == str(target_pbid).split("|")[0]), isoforms[0])
    _pbid, exons, chrom, strand = t_rec
    chrom = _normalize_chrom(chrom, ref)
    if chrom not in ref:
        st.error(f"Chromosome {chrom} not in FASTA (after normalization)."); return

    chseq = ref[chrom].seq
    seq   = "".join(str(chseq[int(s)-1:int(e)]) for (s, e) in sorted(exons))
    if strand == "-":
        seq = str(Seq(seq).reverse_complement())

    # optional trim for folding
    if FAST_MODE and len(seq) > FOLD_LEN_CAP:
        mid  = len(seq)//2
        half = FOLD_LEN_CAP//2
        seq  = seq[max(0, mid-half):mid+half]

    # ---- Fold (cache by gene|pbid|len|k|cap|FAST_MODE)
    key   = f"{gene}|{_pbid}|{len(seq)}|{k}|{FOLD_LEN_CAP}|{FAST_MODE}"
    cache = st.session_state.setdefault("fold_cache", {})
    if key in cache:
        struct, mfe = cache[key]
    else:
        struct, mfe = rnafold_structure(seq.replace("T","U"))
        cache[key]  = (struct, mfe)

    st.caption(f"RNAfold MFE ≈ {mfe} kcal/mol on stitched cDNA of target isoform.")

    # ---- Score windows (structure-first), then uniqueness only on preselected
    seq_U  = seq.replace("T", "U")
    ranked = _rank_windows_fast(seq_U, struct, k, preselect=PRESELECT_N)

    # build other isoform sequences (RNA alphabet) for uniqueness
    other_iso = []
    try:
        for (pb, exs, ch, stn) in isoforms:
            if pb == _pbid:
                continue
            ch2 = _normalize_chrom(ch, ref)
            if ch2 not in ref:
                continue
            s2 = "".join(str(ref[ch2].seq[int(a)-1:int(b)]) for (a, b) in sorted(exs))
            if stn == "-":
                s2 = str(Seq(s2).reverse_complement())
            other_iso.append(s2.replace("T", "U"))
    except Exception:
        pass

    if not ranked.empty:
        # ---- prepare the columns expected by uniqueness_vs_isoforms()
        ranked_tmp = ranked.rename(columns={"sequence_RNA": "sequence"}).copy()

        # base score the uniqueness routine will down-weight
        ranked_tmp["score"] = (
            ranked_tmp["score_struct"]
            if "score_struct" in ranked_tmp.columns else ranked_tmp.get("open_frac", 0.0)
        )

        # it also expects "openness" (we use our open_frac)
        ranked_tmp["openness"] = ranked_tmp.get("open_frac", 0.0)

        # run uniqueness against other isoforms (RNA alphabet sequences)
        ranked_u = uniqueness_vs_isoforms(ranked_tmp, other_iso)

        # restore our naming for downstream display
        ranked = ranked_u.rename(columns={"sequence": "sequence_RNA"})

        cand = ranked.copy()
        cand["GC_frac"]      = cand["sequence_RNA"].apply(lambda s: round((s.count("G")+s.count("C"))/len(s), 3))
        cand["Tm_Wallace"]   = cand["sequence_RNA"].apply(lambda s: wallace_tm(s.replace("U","T")))
        cand["ASO_DNA_5to3"] = cand["sequence_RNA"].apply(lambda s: antisense_from_dna(s.replace("U","T")))
        sort_cols = ["final_score","score_struct","open_frac"] if "final_score" in cand.columns else ["score_struct","open_frac"]
        cand = cand.sort_values(sort_cols, ascending=[False]*len(sort_cols)).head(TOP_N_ASOS).reset_index(drop=True)

        # plots
        # ---- Plots: pairing depth + gray structures with colored ASOs
        fig_depth = plot_pairing_depth_line(struct, cand)
        st.plotly_chart(fig_depth, use_container_width=True)

        # convert end (exclusive) -> inclusive index and clamp to [0, n-1]
        n = len(struct)
        spans = []
        for r in cand.itertuples():
            s = max(0, int(r.start))
            e = min(n - 1, int(r.end) - 1)  # <-- fix off-by-one & clamp
            if e > s:
                spans.append((s, e))

        # interactive structure plots
        st.plotly_chart(arc_plot_interactive(struct, spans),    use_container_width=True)
        st.plotly_chart(circle_plot_interactive(struct, spans), use_container_width=True)

        st.subheader(f"Top {len(cand)} ASO candidates")
        show = cand[["start","end","ASO_DNA_5to3","Tm_Wallace","GC_frac","open_frac","mean_depth"]]
        st.dataframe(show, use_container_width=True, height=320)
        st.download_button(
            "⬇️ Download ASOs (CSV)",
            cand.to_csv(index=False).encode(),
            file_name=f"{gene}_ASO_candidates_{k}mer.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No suitable windows at current length; try a shorter k or disable FAST_MODE temporarily.")



# ------------------------------
# Page registry + navigation
# ------------------------------
from importlib.util import spec_from_file_location, module_from_spec
import inspect

ROOT = Path(__file__).parent.resolve()

def _find_page_file(filename: str) -> Path:
    """
    Find the page file irrespective of where you placed it:
    - ./<filename>
    - ./pages/<filename>
    - any subfolder match (last resort)
    """
    candidates = [
        ROOT / filename,
        ROOT / "pages" / filename,
    ]
    # last-resort recursive search (cheap for small repos)
    candidates += list(ROOT.glob(f"**/{filename}"))

    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            pass
    raise FileNotFoundError(f"Page file not found: {filename}")

def _import_run_from(filename: str):
    """Import the module at filename and call its run(...) with flexible signature."""
    path = _find_page_file(filename)
    spec = spec_from_file_location(path.stem, str(path))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    if not hasattr(mod, "run"):
        raise AttributeError(f"{filename} does not define a run() function")

    fn = mod.run
    sig = inspect.signature(fn)
    kwargs = {}
    # pass supported optional params if present
    if "add_to_shortlist_fn" in sig.parameters:
        kwargs["add_to_shortlist_fn"] = add_to_shortlist
    if "REF_DIR" in sig.parameters:
        kwargs["REF_DIR"] = REF_DIR
    if "DATA_DIR" in sig.parameters:
        kwargs["DATA_DIR"] = DATA_DIR

    return fn(**kwargs)

#def _page_link(label: str, filename: str, icon: str = "📄"):
#    """Show a working page_link regardless of where the file lives."""
#    try:
#        path = _find_page_file(filename)
#        rel = path.relative_to(ROOT)
#        st.page_link(str(rel).replace("\\", "/"), label=label, icon=icon, use_container_width=True)
 #   except Exception:
 #       # fall back to showing a disabled note if not found
 #       st.caption(f"{icon} {label} (missing: {filename})")

# Registry
PAGES = {
    "Home":           page_home,
    "SpinalCord Specificity": lambda: _import_run_from("02_spinal.py"),
    "Pain Genetics":          lambda: _import_run_from("03_pain.py"),
    "DRG Cluster Markers":    lambda: _import_run_from("04_drg.py"),
    "Shortlist":              lambda: _import_run_from("05_shortlist.py"),
    "ASO Design":             page_aso_design
}

with st.sidebar:
    st.subheader("Navigation")
    choice = st.radio("", list(PAGES.keys()), index=0)

    # (optional) a thin separator if you like
    # st.divider()

    st.caption("Shortlist (designable): " + shortlist_preview_text())

# Router
PAGES[choice]()