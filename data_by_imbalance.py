import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO

AA_VALID = set(list("ACDEFGHIKLMNPQRSTVWYXBZJUO"))

def clean_seq(s: str) -> str:
    s = re.sub(r"\s+", "", str(s).upper())
    return "".join([c if c in AA_VALID else "X" for c in s])

# CLASS2ID = {
#     "minor capsid": 0,
#     "tail fiber": 1,
#     "major tail": 2,
#     "portal": 3,
#     "minor tail": 4,
#     "baseplate": 5,
#     "major capsid": 6,
# }
CLASS2ID ={
    "pvp":1,
    "non-pvp":0
}

def map_label(x: str) -> int:
    k = str(x).strip().lower()
    k = k.replace("#", "")
    if k not in CLASS2ID:
        print(f"[WARN] '{x}'")
        return CLASS2ID.get(k, 0)
    return CLASS2ID[k]

class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.labels = [map_label(l) for l in df["label"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = clean_seq(self.df.iloc[idx]["sequence"])
        return seq, torch.tensor(self.labels[idx], dtype=torch.long)

class ESM2Encoder(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        emb = outputs.last_hidden_state.mean(dim=1)
        return emb

@torch.no_grad()
def encode_sequence(seq, tokenizer, model, device, max_len=1022):
    inputs = tokenizer(
        [seq],
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    emb = model(**inputs)  # [1, hidden_size]
    return emb.squeeze(0).cpu().numpy()

def fasta_to_csv(fasta_file, csv_file):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    data = []
    for record in records:
        label = record.description.split()[1]
        data.append({
            "accession": record.id,
            "sequence": str(record.seq),
            "label": label
        })
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"[OK] : {csv_file}")

def process_file(in_csv, out_csv, tokenizer, model, device):
    df = pd.read_csv(in_csv)
    hidden_size = model.base.config.hidden_size
    mat = np.full((len(df), hidden_size), np.nan, dtype=np.float32)
    labels = [map_label(lab) for lab in df["label"]]

    for i, (acc, seq) in tqdm(enumerate(zip(df["accession"], df["sequence"])), total=len(df), desc=f"Encoding {os.path.basename(in_csv)}"):
        seq_clean = clean_seq(str(seq))
        try:
            vec = encode_sequence(seq_clean, tokenizer, model, device)
            mat[i, :] = vec
        except Exception as e:
            print(f"[WARN] {acc}: {e}")

    feat_cols = [f"d{i}" for i in range(hidden_size)]
    out = pd.DataFrame({"accession": df["accession"], "label": labels})
    for j, c in enumerate(feat_cols):
        out[c] = mat[:, j]

    out_dir = os.path.dirname(out_csv)
    os.makedirs(out_dir, exist_ok=True)

    out.to_csv(out_csv, index=False)
    print(f"[OK] {out_csv}, shape={out.shape}")

# ===== 主函数 =====
def main():
    ap = argparse.ArgumentParser("ESM2")
    ap.add_argument("--model-name", default="ESM2_650M")
    ap.add_argument("--data-root", default="imbalance_data/IR_9")
    ap.add_argument("--out-root", default="outputs_imbalance/IR_9")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Info] : {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModel.from_pretrained(args.model_name)
    model = ESM2Encoder(base_model).to(device)
    model.eval()

    fasta_files = {"train": "train.fasta", "valid": "val.fasta", "test": "test.fasta"}
    for split, fasta_file in fasta_files.items():
        in_fasta = os.path.join(args.data_root, fasta_file)
        out_csv = os.path.join(args.out_root, split, "embeddings.csv")
        temp_csv = os.path.join(args.out_root, split, "temp.csv")

        if os.path.exists(in_fasta):
            fasta_to_csv(in_fasta, temp_csv)
            process_file(temp_csv, out_csv, tokenizer, model, device)
        else:
            print(f"[Skip] {in_fasta}")

if __name__ == "__main__":
    main()
