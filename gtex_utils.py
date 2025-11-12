# gtex_utils.py
from pathlib import Path
import os, io, gzip, tempfile, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.Seq import Seq

P = {
    "sup1a":       "data/raw/SupplementaryTable_1A.xlsx",
    "sup1b":       "data/raw/SupplementaryTable_1B.xlsx",
    "gff":         "data/raw/LRS_hDRG_clustered.aligned.collapsed.gff",
    "fasta":       "data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
    "gtf":         "data/reference/Homo_sapiens.GRCh38.115.chr.gtf",
    "gtex_median": "data/reference/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_median_tpm.gct.gz",
}

def load_sup1a(path):
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_excel(path)
    ren = {}
    if "isoform" in df.columns: ren["isoform"] = "pbid"
    if "_associated_gene" in df.columns: ren["_associated_gene"] = "gene_name"
    if "associated_gene" in df.columns: ren["associated_gene"] = "gene_name"
    df = df.rename(columns=ren)
    tpm_cols = [c for c in df.columns if c.startswith("FL_TPM.flnc.") and not c.endswith("_log10")]
    keep = [c for c in ["pbid","gene_name","structural_category","coding","predicted_NMD"] if c in df.columns]
    df = df[keep + tpm_cols].copy()
    if "gene_name" in df.columns:
        df["gene_name"] = df["gene_name"].astype(str).str.upper()
    return df

def load_sup1b(path):
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_excel(path)

def parse_attrs(attr_str: str) -> dict:
    out = {}
    for field in attr_str.strip().strip(";").split(";"):
        field = field.strip()
        if not field:
            continue
        if "=" in field:
            k, v = field.split("=", 1)
        else:
            parts = field.split(" ", 1)
            if len(parts) != 2:
                continue
            k, v = parts
        out[k.strip()] = v.strip().strip('"')
    return out

def parse_gff_fast(gff_path: str):
    """
    Universal GFF parser that supports PacBio-collapsed or standard annotations.
    - Accepts feature: transcript, mRNA, or gene (with exons)
    - Handles ID=PB.17754.6;Parent=...;gene=...
    - Builds transcript entries even if exons come first
    Returns dict: { 'PB.17754.6': {'chr': '1', 'strand': '+', 'exons': [(s,e), ...]} }
    """
    def _core(x: str) -> str:
        # keep PB.17754.6 intact; just remove any |stuff if present
        return x.split("|")[0] if x else x

    tx = {}
    with open(gff_path, "r") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, src, feat, start, end, score, strand, phase, attrs = parts
            start, end = int(start), int(end)
            a = parse_attrs(attrs)
            tx_id = a.get("transcript_id") or a.get("ID") or a.get("Parent") or a.get("gene_id")
            if not tx_id:
                continue
            tx_id = _core(tx_id)
            if feat in ("transcript", "mRNA", "gene"):
                tx.setdefault(tx_id, {"chr": chrom, "strand": strand, "exons": []})
            elif feat == "exon":
                parent = a.get("transcript_id") or a.get("Parent") or a.get("ID")
                parent = _core(parent)
                tx.setdefault(parent, {"chr": chrom, "strand": strand, "exons": []})
                tx[parent]["exons"].append((start, end))
    for v in tx.values():
        v["exons"].sort(key=lambda x: x[0])
    return tx

    

def load_genome_dict_normalized(fasta_path: str):
    out = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        name = rec.id
        if name.startswith("chr"):
            name = name[3:]
        if name in ("M", "chrM", "mitochondrion_genome"):
            name = "MT"
        out[name] = rec
    return out

def _read_gct_median_tpm(gct_gz_path: str) -> pd.DataFrame:
    with gzip.open(gct_gz_path, "rb") as fh:
        df = pd.read_csv(io.BytesIO(fh.read()), sep="\t", skiprows=2)
    cols = list(df.columns)
    cols[0], cols[1] = "Name", "Description"
    df.columns = cols
    tissue_cols = cols[2:]
    genes = df["Description"].astype(str)
    mat = df[tissue_cols].copy()
    mat.index = genes
    mat = mat.groupby(level=0).mean()
    mat = mat.T
    mat.index.name = "tissue"
    return mat

def load_gtex_median(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return _read_gct_median_tpm(path)

def fetch_gene_isoforms(transcripts: dict, sup1a: pd.DataFrame, gene_name: str):
    res = []
    GN = gene_name.upper()
    if "gene_name" not in sup1a.columns or "pbid" not in sup1a.columns:
        return res
    for pbid in sup1a.loc[sup1a["gene_name"] == GN, "pbid"].astype(str):
        key = pbid.split("|")[0]
        if key in transcripts:
            d = transcripts[key]
            res.append((pbid, d["exons"], d["chr"], d["strand"]))
    return res

def gene_isoform_summary(sup1a: pd.DataFrame, gene_name: str) -> pd.DataFrame:
    GN = gene_name.upper()
    sub = sup1a[sup1a["gene_name"] == GN].copy()
    if sub.empty: return sub
    tpm_cols = [c for c in sub.columns if c.startswith("FL_TPM.flnc.")]
    sub["TPM_mean"]     = sub[tpm_cols].mean(axis=1, skipna=True)
    sub["rep_support"]  = (sub[tpm_cols] > 0).sum(axis=1)
    if "structural_category" in sub.columns:
        is_novel = sub["structural_category"].astype(str).str.contains("novel", case=False, na=False).astype(int)
        sub["drg_specificity_score"] = sub["TPM_mean"] * (1 + is_novel)
    else:
        sub["drg_specificity_score"] = sub["TPM_mean"]
    cols = ["pbid","gene_name","structural_category","coding","predicted_NMD"]
    cols = [c for c in cols if c in sub.columns]
    return sub[cols + tpm_cols + ["TPM_mean","rep_support","drg_specificity_score"]] \
             .sort_values("drg_specificity_score", ascending=False)


# gtex_utils.py
def rnafold_structure(seq: str):
    """
    Return (dot_bracket, mfe_kcalmol) for an RNA sequence.
    Prefers ViennaRNA Python bindings (works on Streamlit Cloud with pip).
    Falls back to RNAfold CLI if present. Otherwise returns open structure.
    """
    import numpy as np

    def _open(n):
        return "." * n, np.nan

    if not seq:
        return _open(0)

    s = seq.replace("T", "U")

    # 1) Try ViennaRNA Python bindings (pip package: viennarna -> module name: RNA)
    try:
        import RNA  # from viennarna
        # Fast single-sequence MFE
        ss, mfe = RNA.fold(s)          # returns (structure, mfe)
        if len(ss) != len(s):
            ss = "." * len(s)
        return ss, float(mfe)
    except Exception:
        pass

    # 2) Try ViennaRNA CLI if present
    try:
        import shutil, subprocess, tempfile
        if shutil.which("RNAfold") is not None:
            with tempfile.NamedTemporaryFile("w", delete=False) as fh:
                fh.write(">t\n" + s + "\n")
                tmp = fh.name
            out = subprocess.getoutput(f"RNAfold --noPS < {tmp}")
            lines = [l for l in out.strip().splitlines() if l.strip()]
            struct = lines[-1].split()[0] if lines else "." * len(s)
            try:
                mfe = float(lines[-1].split("(")[-1].split(")")[0])
            except Exception:
                mfe = np.nan
            if len(struct) != len(s):
                struct = "." * len(s)
            return struct, mfe
    except Exception:
        pass

    # 3) Last resort: fully open structure
    return _open(len(s))
    

def find_open_windows(seq, struct, win=18, cutoff=0.80):
    rows = []
    for i in range(0, len(seq)-win+1):
        w = struct[i:i+win]
        open_frac = w.count('.')/win
        if open_frac >= cutoff:
            sseq = seq[i:i+win]
            gc = 100.0*(sseq.count('G')+sseq.count('C'))/win
            bad = any(b*5 in sseq for b in "AUGC")
            rows.append({"start":i,"end":i+win,"sequence":sseq,"openness":open_frac,"GC":gc,"bad_poly":bad})
    df = pd.DataFrame(rows)
    if df.empty: return df
    df = df[~df["bad_poly"]].copy()
    df["gc_penalty"] = (1 - abs(df["GC"]-45)/45).clip(lower=0)
    df["score"] = df["openness"] * df["gc_penalty"]
    return df.sort_values(["score","openness"], ascending=False).reset_index(drop=True)

def uniqueness_vs_isoforms(cands_df, other_seqs):
    if cands_df.empty: return cands_df
    def hits(s):
        return sum((s in o) for o in other_seqs if o)
    out = cands_df.copy()
    out["isoform_hits_elsewhere"] = out["sequence"].apply(hits)
    out["final_score"] = out["score"]/(1+out["isoform_hits_elsewhere"])
    return out.sort_values(["final_score","openness"], ascending=False)

def wallace_tm(seq: str):
    a = seq.count("A"); t = seq.count("U"); g = seq.count("G"); c = seq.count("C")
    return 2*(a+t) + 4*(g+c)

def plot_replicate_tpms_matplotlib(gene_table: pd.DataFrame, gene_name: str):
    if gene_table.empty: return None
    tpm_cols = [c for c in gene_table.columns if c.startswith("FL_TPM.flnc.")]
    long = gene_table.melt(id_vars=["pbid"], value_vars=tpm_cols, var_name="replicate", value_name="TPM")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios":[2,1]})
    for rep in long["replicate"].unique():
        sub = long[long["replicate"] == rep]
        ax[0].scatter(sub["pbid"], sub["TPM"], s=22, label=rep, alpha=0.85)
    ax[0].set_title(f"{gene_name} — isoform TPM per replicate")
    ax[0].set_ylabel("TPM"); ax[0].tick_params(axis='x', rotation=90); ax[0].legend()
    long.boxplot(column="TPM", by="replicate", ax=ax[1])
    ax[1].set_title("replicate TPM distribution"); ax[1].set_ylabel("TPM")
    plt.suptitle(""); plt.tight_layout()
    return fig

def _merge_intervals(ivals):
    if not ivals: return []
    ivals = sorted((int(s), int(e)) for s, e in ivals)
    out = [list(ivals[0])]
    for s, e in ivals[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]

def load_gtf_regions_for_gene(gtf_path: str, gene_name: str, chrom_hint: str=None):
    gp = Path(gtf_path)
    if not gp.exists(): return None
    GN = gene_name.upper()
    regs = {"5UTR":[], "CDS":[], "3UTR":[]}
    with open(gtf_path, "r") as fh:
        for line in fh:
            if not line or line.startswith("#"): continue
            chrom, src, feat, start, end, score, strand, phase, attrs = line.rstrip("\n").split("\t")
            a = parse_attrs(attrs)
            gname = (a.get("gene_name") or a.get("gene_id") or "").upper()
            if gname != GN: continue
            if chrom_hint and chrom != chrom_hint: continue
            start, end = int(start), int(end)
            if feat in ("three_prime_UTR","3UTR"):
                regs["3UTR"].append((start,end))
            elif feat in ("five_prime_UTR","5UTR"):
                regs["5UTR"].append((start,end))
            elif feat == "CDS":
                regs["CDS"].append((start,end))
    for k in regs: regs[k].sort(key=lambda x: x[0])
    if sum(len(v) for v in regs.values()) == 0: return None
    return regs

def plot_isoform_schematic_fast(isoforms, target_pbid=None, gtf_regions=None, title="Isoform schematic (fast)"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    def _add_broken_barh(ax, xranges, yrange, facecolors=None, edgecolors=None, linewidth=0.8):
        ax.broken_barh(xranges, yrange, facecolors=facecolors, edgecolors=edgecolors, linewidth=linewidth)

    if not isoforms:
        return None

    exon_lists = [ex for _, ex, _, _ in isoforms]
    xmin = min(s for ex in exon_lists for s, _ in ex)
    xmax = max(e for ex in exon_lists for _, e in ex)

    if gtf_regions:
        gtf_regions = {k: _merge_intervals(v) for k, v in gtf_regions.items()}

    fig, ax = plt.subplots(figsize=(12, max(2, 0.55 * len(isoforms))))

    if gtf_regions:
        for (s, e) in gtf_regions.get("5UTR", []):
            ax.axvspan(s, e, color="#9ecae1", alpha=0.16)
        for (s, e) in gtf_regions.get("CDS", []):
            ax.axvspan(s, e, color="#a1d99b", alpha=0.12)
        for (s, e) in gtf_regions.get("3UTR", []):
            ax.axvspan(s, e, color="#fdae6b", alpha=0.16)

    y = 0.0; h = 0.75
    tcore = target_pbid.split("|")[0] if target_pbid else None
    for pbid, exons, chrom, strand in isoforms:
        xranges = [(int(s), int(e - s)) for (s, e) in exons if e > s]
        _add_broken_barh(ax, xranges, (y, h), facecolors="#A6D96A", edgecolors="#1B9E77", linewidth=0.8)
        if (pbid.split("|")[0] == tcore):
            ax.broken_barh([(xmin, xmax - xmin)], (y, h), facecolors="none", edgecolors="#1B9E77", linewidth=1.6)
            ax.text(xmin, y + h + 0.05, pbid, fontsize=8, va="bottom")
        y += 0.95

    handles = [
        mpatches.Patch(facecolor="#66C2A5", edgecolor="#1B9E77", label="Target isoform"),
        mpatches.Patch(facecolor="#A6D96A", edgecolor="none",    label="Other isoforms")
    ]
    if gtf_regions:
        handles += [
            mpatches.Patch(facecolor="#9ecae1", alpha=0.16, label="5'UTR"),
            mpatches.Patch(facecolor="#a1d99b", alpha=0.12, label="CDS"),
            mpatches.Patch(facecolor="#fdae6b", alpha=0.16, label="3'UTR"),
        ]
    ax.legend(handles=handles, frameon=False, ncol=5, fontsize=8)
    ax.set_xlim(xmin, xmax)
    ax.set_yticks([])
    ax.set_xlabel("Genomic position")
    ax.set_title(title)
    fig.tight_layout()
    return fig