import os, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_multi import ProteinClassifier


# =========================
# Dataset
# =========================
class TestDataset(Dataset):
    def __init__(self, df, feat_prefix="d"):
        self.df = df.reset_index(drop=True)
        self.feat_cols = sorted(
            [c for c in df.columns if c.startswith(feat_prefix)],
            key=lambda x: int(x[len(feat_prefix):]) if x[len(feat_prefix):].isdigit() else x
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = row[self.feat_cols].to_numpy(dtype=np.float32)
        return torch.tensor(x, dtype=torch.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="outputs_case_study/case_study/embeddings.csv")
    ap.add_argument("--feat-prefix", default="d")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-path", default="./spilt_by_time_multi/best.pt")
    ap.add_argument("--out-csv", default="outputs_case_study/multi_prediction_results_no.csv")
    args = ap.parse_args()

    device = torch.device(args.device)

    test_df = pd.read_csv(args.test)
    feat_cols = [c for c in test_df.columns if c.startswith(args.feat_prefix)]
    D = len(feat_cols)
    num_classes = 7

    print(f"[Info] D={D}, {num_classes}")

    classifier = ProteinClassifier(dim=D, num_classes=num_classes).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    classifier.load_state_dict(checkpoint["model"])
    classifier.eval()
    print(f"[Info] {args.model_path}")

    test_loader = DataLoader(TestDataset(test_df, feat_prefix=args.feat_prefix),
                             batch_size=16,
                             shuffle=False)

    all_logits = []

    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            logits = classifier(x)  # (B, num_classes)
            all_logits.append(logits.cpu().numpy())

    all_logits = np.concatenate(all_logits)  # (N, num_classes)
    preds = all_logits.argmax(axis=1)
    probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
    max_probs = probs.max(axis=1)

    out_df = test_df.copy()
    out_df["pred_label"] = preds
    out_df["pred_prob"] = max_probs

    class_names = {
        0: "minor capsid",
        1: "tail fiber",
        2: "major tail",
        3: "portal",
        4: "minor tail",
        5: "baseplate",
        6: "major capsid",
    }
    out_df["class_name"] = out_df["pred_label"].map(class_names)

    for i in range(num_classes):
        out_df[f"prob_class{i}"] = probs[:, i]

    out_df.to_csv(args.out_csv, index=False)
    print(f"[OK] {args.out_csv}")


if __name__ == "__main__":
    main()
