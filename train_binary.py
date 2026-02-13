import os, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report, matthews_corrcoef
from model_binary import ProteinClassifier


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

class Generator(nn.Module):
    def __init__(self, z_dim, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, x_dim)
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_mlp_wgan_gp(real_feats, device, epochs=1000, batch_size=64,
                      noise_dim=32, lambda_gp=10):
    x_dim = real_feats.shape[1]
    G = Generator(noise_dim, x_dim).to(device)
    D = Discriminator(x_dim).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

    real_tensor = torch.tensor(real_feats, dtype=torch.float32).to(device)

    for epoch in range(1, epochs + 1):
        # ---- Train Discriminator ----
        idx = torch.randint(0, len(real_tensor), (batch_size,))
        real_batch = real_tensor[idx]

        z = torch.randn(batch_size, noise_dim).to(device)
        fake_batch = G(z).detach()

        real_score = D(real_batch)
        fake_score = D(fake_batch)

        # Gradient penalty
        alpha = torch.rand(batch_size, 1).to(device)
        interp = alpha * real_batch + (1 - alpha) * fake_batch
        interp.requires_grad_(True)
        grad = torch.autograd.grad(
            outputs=D(interp).sum(),
            inputs=interp,
            create_graph=True
        )[0]
        gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()

        loss_D = -(real_score.mean() - fake_score.mean()) + lambda_gp * gp

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ---- Train Generator ----
        if epoch % 5 == 0:
            z = torch.randn(batch_size, noise_dim).to(device)
            fake_batch = G(z)
            fake_score = D(fake_batch)
            loss_G = -fake_score.mean()

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        if epoch % 200 == 0 or epoch == 1:
            print(f"[GAN Epoch {epoch}/{epochs}] D_loss={loss_D.item():.4f}")

    return G


# =========================
# Metrics
# =========================
def compute_metrics(all_logits, all_targets):
    preds = all_logits.argmax(axis=1)
    acc = (preds == all_targets).mean()
    f1_w = f1_score(all_targets, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_targets, preds)
    mcc = matthews_corrcoef(all_targets, preds)
    report = classification_report(all_targets, preds, digits=4, output_dict=True)

    labels = sorted(np.unique(all_targets))
    class_f1 = {int(l): report[str(l)]["f1-score"] for l in labels}
    class_acc = {}
    for i, l in enumerate(labels):
        tp = cm[i, i]
        total = cm[i].sum()
        class_acc[int(l)] = tp / total if total > 0 else 0.0

    return acc, f1_w, report, cm, mcc, class_f1, class_acc


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train(training)
    all_logits, all_targets, total_loss, n_samples = [], [], 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if training:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)

    all_logits = np.concatenate(all_logits)
    all_targets = np.concatenate(all_targets)
    acc, f1_w, report, cm, mcc, class_f1, class_acc = compute_metrics(all_logits, all_targets)
    return total_loss / n_samples, acc, f1_w, report, cm, mcc, class_f1, class_acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="outputs_imbalance/IR_9/train/embeddings.csv")
    ap.add_argument("--dev", default="outputs_imbalance/IR_9/valid/embeddings.csv")
    ap.add_argument("--test", default="outputs_imbalance/IR_9/test/embeddings.csv")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--feat-prefix", default="d")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-dir", default="./spilt_by_imbalance_9")

    ap.add_argument("--use-wgan", action="store_true", help="是否启用 WGAN-GP 数据增强")
    ap.add_argument("--z-dim", type=int, default=32)
    ap.add_argument("--wgan-epochs", type=int, default=1000)
    ap.add_argument("--wgan-batch", type=int, default=64)
    ap.add_argument("--k-classes", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    train_df = pd.read_csv(args.train)
    dev_df = pd.read_csv(args.dev)
    test_df = pd.read_csv(args.test)

    feat_cols = [c for c in train_df.columns if c.startswith(args.feat_prefix)]
    D = len(feat_cols)
    print(f"[Info] embedding D={D}")

    labels, counts = np.unique(train_df[args.label_col], return_counts=True)
    label_counts = sorted(list(zip(labels, counts)), key=lambda x: x[1])
    print("[Info]", label_counts)

    if args.use_wgan:
        print(f"[Info]{args.k_classes} ")
        k_least_labels = [x[0] for x in label_counts[:args.k_classes]]

        augmented_dfs = [train_df]
        for cls, count in zip(labels, counts):
            if cls not in k_least_labels:
                continue

            target_count = 3000
            n_generate = target_count - count
            if n_generate <= 0:
                continue

            print(f"[Info] {cls} {n_generate}")
            cls_feats = train_df[train_df[args.label_col] == cls][feat_cols].to_numpy()
            G = train_mlp_wgan_gp(cls_feats, device=device,
                                  epochs=args.wgan_epochs,
                                  batch_size=args.wgan_batch,
                                  noise_dim=args.z_dim)

            z = torch.randn(n_generate, args.z_dim).to(device)
            fake_feats = G(z).detach().cpu().numpy()
            fake_df = pd.DataFrame(fake_feats, columns=feat_cols)
            fake_df[args.label_col] = cls
            augmented_dfs.append(fake_df)

        train_df = pd.concat(augmented_dfs, ignore_index=True)
        print(f"[Info] {train_df.shape}")
    else:
        print("[Info]")

    train_loader = DataLoader(InlineEmbeddingDataset(train_df), batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad)
    dev_loader = DataLoader(InlineEmbeddingDataset(dev_df), batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad)
    test_loader = DataLoader(InlineEmbeddingDataset(test_df), batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad)

    num_classes = len(np.unique(train_df[args.label_col]))
    classifier = ProteinClassifier(dim=D, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=4e-4)

    best_f1, best_path = -1, os.path.join(args.save_dir, "best.pt")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1, _, _, _, _, _ = run_epoch(classifier, train_loader, criterion, optimizer, device)
        dv_loss, dv_acc, dv_f1, _, _, _, _, _ = run_epoch(classifier, dev_loader, criterion, None, device)
        print(f"[Epoch {epoch}] train_f1={tr_f1:.4f} dev_f1={dv_f1:.4f}")
        if dv_f1 > best_f1:
            best_f1 = dv_f1
            torch.save({"model": classifier.state_dict()}, best_path)

    classifier.load_state_dict(torch.load(best_path)["model"])
    te_loss, te_acc, te_f1, te_report, te_cm, te_mcc, te_class_f1, te_class_acc = run_epoch(
        classifier, test_loader, criterion, None, device
    )

    print("\n========== [Test Results] ==========")
    print(f"Overall Accuracy: {te_acc:.4f}")
    print(f"Weighted F1-score: {te_f1:.4f}")
    print(f"MCC: {te_mcc:.4f}")
    print("Confusion Matrix:\n", te_cm)

    print("\nPer-class Metrics:")
    for cls in sorted(te_class_f1.keys()):
        print(f"Class {cls}: acc={te_class_acc[cls]:.4f}, f1={te_class_f1[cls]:.4f}")

    print("\nDetailed classification report:")
    print(pd.DataFrame(te_report).transpose())


if __name__ == "__main__":
    main()
