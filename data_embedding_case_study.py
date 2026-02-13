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

class ESM2Encoder(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        emb = outputs.last_hidden_state.mean(dim=1)  # 平均池化
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

def process_file(in_csv, out_csv, tokenizer, model, device):
    df = pd.read_csv(in_csv)
    hidden_size = model.base.config.hidden_size

    preserve_cols = [col for col in df.columns if col != "protein_sequence"]

    mat = np.full((len(df), hidden_size), np.nan, dtype=np.float32)

    for i, seq_row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Encoding {os.path.basename(in_csv)}"
    ):
        seq = seq_row["protein_sequence"]
        seq_clean = clean_seq(str(seq))

        acc_info = seq_row.get("id", seq_row.get("accession", f"Row {i}"))

        try:
            vec = encode_sequence(seq_clean, tokenizer, model, device)
            mat[i, :] = vec
        except Exception as e:
            print(f"[WARN] {acc_info} : {e}")

    out = df[preserve_cols].copy()

    feat_cols = [f"d{i}" for i in range(hidden_size)]
    for j, c in enumerate(feat_cols):
        out[c] = mat[:, j]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[OK] {out_csv}, shape={out.shape}")


def main():
    ap = argparse.ArgumentParser("ESM2")
    ap.add_argument("--model-name", default="ESM2_650M")
    ap.add_argument("--data-root", default="data_host_prediction")
    ap.add_argument("--out-root", default="data_host_prediction")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Info] {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModel.from_pretrained(args.model_name)
    model = ESM2Encoder(base_model).to(device)
    model.eval()

    files = {
        "RBP": "RBP.csv"
    }

    for split, fname in files.items():
        in_csv = os.path.join(args.data_root, fname)
        out_csv = os.path.join(args.out_root, split, "embeddings.csv")

        if os.path.exists(in_csv):
            process_file(in_csv, out_csv, tokenizer, model, device)
        else:
            print(f"[Skip] {in_csv}")


if __name__ == "__main__":
    main()