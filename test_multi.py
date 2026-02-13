import os, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report, matthews_corrcoef, accuracy_score

from model_multi import ProteinClassifier


# =========================
# Dataset
# =========================
class InlineEmbeddingDataset(Dataset):
    def __init__(self, df, label_col="label", feat_prefix="d"):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.feat_cols = sorted(
            [c for c in df.columns if c.startswith(feat_prefix)],
            key=lambda x: int(x[len(feat_prefix):]) if x[len(feat_prefix):].isdigit() else x
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = row[self.feat_cols].to_numpy(dtype=np.float32)
        y = int(row[self.label_col])
        return {"x": x, "y": y}


def collate_pad(batch):
    xs = torch.tensor([b["x"] for b in batch], dtype=torch.float32)
    ys = torch.tensor([b["y"] for b in batch], dtype=torch.long)
    return xs, ys


# =========================
# Metrics
# =========================
def compute_metrics(all_logits, all_targets):
    preds = all_logits.argmax(axis=1)
    acc = (preds == all_targets).mean()
    f1_w = f1_score(all_targets, preds, average='weighted', zero_division=0)
    f1_per_class = f1_score(all_targets, preds, average=None, zero_division=0)

    acc_per_class = []
    unique_classes = np.unique(all_targets)
    for cls in unique_classes:
        cls_mask = all_targets == cls
        cls_acc = accuracy_score(all_targets[cls_mask], preds[cls_mask]) if np.sum(cls_mask) > 0 else 0.0
        acc_per_class.append(cls_acc)

    cm = confusion_matrix(all_targets, preds)
    mcc = matthews_corrcoef(all_targets, preds)
    report = classification_report(all_targets, preds, digits=4)
    return acc, f1_w, f1_per_class, acc_per_class, report, cm, mcc


def run_epoch(model, loader, criterion, device="cpu"):
    model.eval()
    all_logits, all_targets, total_loss, n_samples = [], [], 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            all_logits.append(logits.detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())
            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
    all_logits = np.concatenate(all_logits)
    all_targets = np.concatenate(all_targets)
    acc, f1_w, f1_per_class, acc_per_class, report, cm, mcc = compute_metrics(all_logits, all_targets)
    return total_loss / n_samples, acc, f1_w, f1_per_class, acc_per_class, report, cm, mcc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="outputs_smi/multi_esm2_encode_40/test/embeddings.csv")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--feat-prefix", default="d")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-path", default="./weight/spilt_by_smi_multi_40_wgan/best.pt")
    args = ap.parse_args()

    device = torch.device(args.device)

    test_df = pd.read_csv(args.test)
    feat_cols = [c for c in test_df.columns if c.startswith(args.feat_prefix)]
    D = len(feat_cols)
    num_classes = len(np.unique(test_df[args.label_col]))
    print(f"[Info]D={D},{num_classes}")

    # ====== 加载模型 ======
    classifier = ProteinClassifier(dim=D, num_classes=num_classes).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    classifier.load_state_dict(checkpoint["model"])
    print(f"[Info] {args.model_path}")

    # ====== 构建 DataLoader ======
    test_loader = DataLoader(
        InlineEmbeddingDataset(test_df, label_col=args.label_col, feat_prefix=args.feat_prefix),
        batch_size=8,
        shuffle=False,
        collate_fn=collate_pad
    )

    criterion = nn.CrossEntropyLoss()

    te_loss, te_acc, te_f1, te_f1_per_class, te_acc_per_class, te_report, te_cm, te_mcc = run_epoch(
        classifier, test_loader, criterion, device
    )


if __name__ == "__main__":
    main()
