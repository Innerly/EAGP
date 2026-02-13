import torch
import torch.nn as nn

def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu":
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f"Unsupported activation: {name}")

class BinaryClassHead(nn.Module):
    def __init__(self, dim: int, num_classes: int = 2, hidden: int = 512,
                 dropout: float = 0.3, act_name: str = "gelu"):
        super().__init__()
        act = get_activation(act_name)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ProteinClassifier(nn.Module):

    def __init__(self, dim=640, num_classes=2, dropout=0.3, act_name="gelu"):
        super().__init__()
        self.classifier = BinaryClassHead(
            dim=dim, num_classes=num_classes, dropout=dropout, act_name=act_name
        )

    def forward(self, x):
        logits = self.classifier(x)
        return logits
