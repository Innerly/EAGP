import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

AA_VALID = set(list("ACDEFGHIKLMNPQRSTVWYXBZJUO"))

def clean_seq(s: str) -> str:
    s = re.sub(r"\s+", "", str(s).upper())
    return "".join([c if c in AA_VALID else "X" for c in s])

CLASS2ID = {
    "minor capsid": 0,
    "tail fiber": 1,
    "major tail": 2,
    "portal": 3,
    "minor tail": 4,
    "baseplate": 5,
    "major capsid": 6,
}

def map_label(x: str) -> int:
    k = str(x).strip().lower()
    return CLASS2ID.get(k, -1)

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
    emb = model(**inputs)
    return emb.squeeze(0).cpu().numpy()

def process_file(in_csv, out_csv, tokenizer, model, device):
    df = pd.read_csv(in_csv)
    print(f"[Info] {in_csv}{len(df)}")

    hidden_size = model.base.config.hidden_size
    mat = np.full((len(df), hidden_size), np.nan, dtype=np.float32)

    labels = []
    if "label" in df.columns:
        labels = [map_label(lab) for lab in df["label"]]
    else:
        labels = [-1] * len(df)

    for i, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df), desc=f"Encoding {os.path.basename(in_csv)}"):
        seq = clean_seq(str(row["sequence"]))
        try:
            vec = encode_sequence(seq, tokenizer, model, device)
            mat[i, :] = vec
        except Exception as e:
            print(f"[WARN]{i+1} : {e}")

    feat_cols = [f"d{i}" for i in range(hidden_size)]
    out = pd.DataFrame({"accession": df["accession"] if "accession" in df.columns else np.arange(len(df))})
    out["label"] = labels
    for j, c in enumerate(feat_cols):
        out[c] = mat[:, j]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[OK]{out_csv}, shape={out.shape}")

def main():
    ap = argparse.ArgumentParser("ESM2")
    ap.add_argument("--model-name", default="ESM2_650M")
    ap.add_argument("--input-csv", default="minor_capsid.csv", help="CSV")
    ap.add_argument("--output-csv", default="outputs/minor_capsid_emb.csv", help="CSV")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Info] : {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModel.from_pretrained(args.model_name)
    model = ESM2Encoder(base_model).to(device)
    model.eval()

    process_file(args.input_csv, args.output_csv, tokenizer, model, device)

if __name__ == "__main__":
    main()
