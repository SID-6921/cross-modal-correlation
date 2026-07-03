"""
Intermediate-cardinality closed-set experiment, addressing the "confound we
have not disentangled" limitation in Section 4.4: the paper's InfoNCE-wins
result at open-vocabulary scale (~5,000 captions) vs. Barlow-Twins-wins at
closed-set scale (10 digit classes) changes three factors at once (target
cardinality, open-vs-closed vocabulary, frozen-vs-trained backbone). This
experiment isolates cardinality alone: real CIFAR-100 images (100 classes,
still closed-set, still a fully-trained-from-scratch small CNN encoder
like Experiment 1 -- NOT frozen/pretrained) paired with a synthetic
per-class signal, one distinct frequency per class. If InfoNCE starts
closing the gap here (at cardinality 100, still closed-set, still trained
backbone), that supports cardinality as the real driver rather than
open-vocabulary-ness or the frozen backbone.

Usage: python run_cardinality_ablation.py <objective> <seed>
  objective: "barlow" or "infonce"
"""
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

OBJECTIVE = sys.argv[1] if len(sys.argv) > 1 else "barlow"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
assert OBJECTIVE in ("barlow", "infonce")

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[cardinality100 {OBJECTIVE} seed {SEED}] Device: {DEVICE}", flush=True)

NUM_CLASSES = 100
SIGNAL_LEN = 64
BASE_FREQS = [3 + 1.5 * c for c in range(NUM_CLASSES)]  # 100 distinct frequencies


def make_signal_for_class(cls, seed_offset=0):
    g = torch.Generator().manual_seed(hash((cls, seed_offset)) % (2**31))
    t = torch.arange(SIGNAL_LEN).float()
    freq = BASE_FREQS[cls]
    phase = torch.rand(1, generator=g).item() * 2 * math.pi
    noise = torch.randn(SIGNAL_LEN, generator=g) * 0.15
    return torch.sin(2 * math.pi * freq * t / SIGNAL_LEN + phase) + noise


class PairedCIFAR100Signal(torch.utils.data.Dataset):
    def __init__(self, cifar_dataset, seed_offset=0):
        self.cifar = cifar_dataset
        self.seed_offset = seed_offset

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        img, label = self.cifar[idx]
        signal = make_signal_for_class(label, seed_offset=self.seed_offset + idx)
        return img, signal, label


class ImageEncoder(nn.Module):
    """Small CNN, trained from scratch (not pretrained/frozen) -- same
    family as Experiment 1's MNIST encoder, scaled slightly for 32x32 RGB."""
    def __init__(self, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 4x4
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, embed_dim),
        )
    def forward(self, x): return self.net(x)


class SignalEncoder(nn.Module):
    def __init__(self, embed_dim=64, signal_len=SIGNAL_LEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(), nn.Linear(32 * (signal_len // 4), 128), nn.ReLU(), nn.Linear(128, embed_dim),
        )
    def forward(self, x): return self.net(x.unsqueeze(1))


def barlow_twins_loss(z_a, z_b, lambda_offdiag=0.005):
    n, d = z_a.shape
    z_a_norm = (z_a - z_a.mean(0)) / (z_a.std(0) + 1e-8)
    z_b_norm = (z_b - z_b.mean(0)) / (z_b.std(0) + 1e-8)
    c = (z_a_norm.T @ z_b_norm) / n
    on_diag = ((torch.diagonal(c) - 1) ** 2).sum()
    off_diag = (c.pow(2).sum() - torch.diagonal(c).pow(2).sum())
    return on_diag + lambda_offdiag * off_diag


def info_nce_loss(z_a, z_b, temperature=0.1):
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = z_a @ z_b.T / temperature
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_b) / 2


def main():
    print(f"\n=== [cardinality100 {OBJECTIVE} seed {SEED}] Loading real CIFAR-100 ===", flush=True)
    transform = transforms.Compose([transforms.ToTensor()])
    train_cifar = datasets.CIFAR100(root="./cifar100_data", train=True, download=True, transform=transform)
    test_cifar = datasets.CIFAR100(root="./cifar100_data", train=False, download=True, transform=transform)

    train_ds = PairedCIFAR100Signal(train_cifar, seed_offset=SEED)
    test_ds = PairedCIFAR100Signal(test_cifar, seed_offset=100000 + SEED)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)

    EMBED_DIM = 64
    img_encoder = ImageEncoder(EMBED_DIM).to(DEVICE)
    sig_encoder = SignalEncoder(EMBED_DIM).to(DEVICE)
    opt = torch.optim.Adam(list(img_encoder.parameters()) + list(sig_encoder.parameters()), lr=1e-3)
    loss_fn = barlow_twins_loss if OBJECTIVE == "barlow" else info_nce_loss

    EPOCHS = 20
    print(f"\n=== [cardinality100 {OBJECTIVE} seed {SEED}] Training, {EPOCHS} epochs, {NUM_CLASSES} classes ===", flush=True)
    t0 = time.time()
    for epoch in range(EPOCHS):
        img_encoder.train(); sig_encoder.train()
        total_loss, n_batches = 0.0, 0
        for imgs, signals, _labels in train_loader:
            imgs, signals = imgs.to(DEVICE), signals.to(DEVICE)
            opt.zero_grad()
            z_a = img_encoder(imgs)
            z_b = sig_encoder(signals)
            loss = loss_fn(z_a, z_b)
            loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1
        print(f"  [cardinality100 {OBJECTIVE} seed {SEED}] epoch {epoch+1}: loss={total_loss/n_batches:.4f} "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)
    train_time = time.time() - t0

    print(f"\n=== [cardinality100 {OBJECTIVE} seed {SEED}] Eval ===", flush=True)
    img_encoder.eval(); sig_encoder.eval()
    with torch.no_grad():
        train_embeds, train_labels = [], []
        for imgs, signals, labels in train_loader:
            z = img_encoder(imgs.to(DEVICE))
            train_embeds.append(z.cpu()); train_labels.append(labels)
        train_embeds = torch.cat(train_embeds); train_labels = torch.cat(train_labels)

        test_embeds, test_labels, test_sig_embeds = [], [], []
        for imgs, signals, labels in test_loader:
            z = img_encoder(imgs.to(DEVICE))
            zs = sig_encoder(signals.to(DEVICE))
            test_embeds.append(z.cpu()); test_sig_embeds.append(zs.cpu()); test_labels.append(labels)
        test_embeds = torch.cat(test_embeds); test_sig_embeds = torch.cat(test_sig_embeds); test_labels = torch.cat(test_labels)

    probe = nn.Linear(EMBED_DIM, NUM_CLASSES).to(DEVICE)
    probe_opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    tr_x, tr_y = train_embeds.to(DEVICE), train_labels.to(DEVICE)
    te_x, te_y = test_embeds.to(DEVICE), test_labels.to(DEVICE)
    for _ in range(100):
        probe_opt.zero_grad()
        loss = F.cross_entropy(probe(tr_x), tr_y)
        loss.backward(); probe_opt.step()
    with torch.no_grad():
        probe_acc = (probe(te_x).argmax(-1) == te_y).float().mean().item()

    n_pool = min(1000, len(test_embeds))
    idx = torch.randperm(len(test_embeds))[:n_pool]
    pool_img = F.normalize(test_embeds[idx], dim=-1)
    pool_sig = F.normalize(test_sig_embeds[idx], dim=-1)
    pool_labels = test_labels[idx]
    sims = pool_img @ pool_sig.T
    top1 = sims.argmax(dim=-1)
    top1_correct = (pool_labels[top1] == pool_labels).float().mean().item()
    top5 = sims.topk(5, dim=-1).indices
    top5_correct = torch.tensor([pool_labels[top5[i]].eq(pool_labels[i]).any() for i in range(n_pool)]).float().mean().item()

    print(f"  [cardinality100 {OBJECTIVE} seed {SEED}] linear_probe={probe_acc:.4f} top1={top1_correct:.4f} top5={top5_correct:.4f} "
          f"(chance: probe={1/NUM_CLASSES:.4f}, top1={1/NUM_CLASSES:.4f}, top5={5/NUM_CLASSES:.4f})", flush=True)

    record = {
        "objective": OBJECTIVE, "seed": SEED, "num_classes": NUM_CLASSES,
        "linear_probe_accuracy": probe_acc, "retrieval_top1_accuracy": top1_correct,
        "retrieval_top5_accuracy": top5_correct, "train_time_s": train_time,
        "chance_probe": 1 / NUM_CLASSES, "chance_top1": 1 / NUM_CLASSES, "chance_top5": 5 / NUM_CLASSES,
    }
    out_file = f"cardinality100_{OBJECTIVE}_seed{SEED}_results.json"
    with open(out_file, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nSaved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
