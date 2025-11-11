import pandas as pd
from pathlib import Path
from Bio import SeqIO
import requests, io, gzip

# Google Drive direct-download links (replace with your share IDs)
GFF_URL = "https://drive.google.com/uc?export=download&id=1Q_uIGt1mvP8Ttnw2S6vMcY1piLlFx4r9"
FASTA_URL = "https://drive.google.com/uc?export=download&id=1nzcs1XW6Nw6ZYucFdfiHIk6KC5p_A0W1"

def download_file(url, local_path):
    """Download file only if not present locally."""
    path = Path(local_path)
    if not path.exists():
        print(f"Downloading {url} → {local_path}")
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
    return path

def load_gff(local_path="data/raw/LRS_hDRG_clustered.aligned.collapsed.gff"):
    path = download_file(GFF_URL, local_path)
    return pd.read_csv(path, sep="\t", comment="#", header=None)

def load_fasta(local_path="data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa", limit=None):
    path = download_file(FASTA_URL, local_path)
    handle = open(path, "r")
    records = SeqIO.parse(handle, "fasta")
    if limit:  # for demo, load only first few sequences
        records = list(r for _, r in zip(range(limit), records))
    return {r.id: str(r.seq) for r in records}