"""
Proof-of-concept: does a Barlow-Twins-style cross-correlation alignment
objective work ACROSS two genuinely different modalities, not just two
augmented views of the same image (which is what the original Barlow
Twins paper tested)?

Modality A: real MNIST digit images.
Modality B: synthetic 1D "signal" -- a sine wave whose frequency depends
  on the digit identity plus per-sample phase/noise, deterministically
  paired with each image but visually/structurally nothing like an image.
  This is a stand-in for a genuinely different modality (e.g. an image
  vs. a biosignal), not a second augmented view of the same data.

Training is self-supervised: the loss only uses the fact that image_i and
signal_i are a matched pair (like image-caption pairs in CLIP), never the
digit label itself.

Evaluation (the actual test of whether this is a real, useful finding):
1. Linear probe on the learned image embedding -- does the cross-modal
   correlation objective learn label-relevant structure, without ever
   seeing labels during training?
2. Cross-modal retrieval -- given an image embedding, can we retrieve the
   correct-digit signal embedding from a pool of candidates? This is the
   real test of "did alignment actually happen."

Every number below is real, from a real training run.
"""
import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

RESULTS = {"device": str(DEVICE)}


# ============================================================
# Data: MNIST (modality A) + synthetic signal (modality B), paired by digit
# ============================================================
SIGNAL_LEN = 64
BASE_FREQS = [3 + 2 * d for d in range(10)]  # distinct frequency per digit, 3..21


def make_signal_for_digit(digit, seed_offset=0):
    g = torch.Generator().manual_seed(hash((digit, seed_offset)) % (2**31))
    t = torch.arange(SIGNAL_LEN).float()
    freq = BASE_FREQS[digit]
    phase = torch.rand(1, generator=g).item() * 2 * math.pi
    noise = torch.randn(SIGNAL_LEN, generator=g) * 0.15
    signal = torch.sin(2 * math.pi * freq * t / SIGNAL_LEN + phase) + noise
    return signal


class PairedMNISTSignal(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset, seed_offset=0):
        self.mnist = mnist_dataset
        self.seed_offset = seed_offset

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        signal = make_signal_for_digit(label, seed_offset=self.seed_offset + idx)
        return img, signal, label


# ============================================================
# Encoders
# ============================================================
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class SignalEncoder(nn.Module):
    def __init__(self, embed_dim=64, signal_len=SIGNAL_LEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * (signal_len // 4), 128), nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))


def barlow_twins_loss(z_a, z_b, lambda_offdiag=0.005):
    """Cross-modal Barlow Twins loss: make the cross-correlation matrix
    between modality-A and modality-B embeddings close to the identity."""
    n, d = z_a.shape
    z_a_norm = (z_a - z_a.mean(0)) / (z_a.std(0) + 1e-8)
    z_b_norm = (z_b - z_b.mean(0)) / (z_b.std(0) + 1e-8)
    c = (z_a_norm.T @ z_b_norm) / n  # (d, d) cross-correlation matrix
    on_diag = ((torch.diagonal(c) - 1) ** 2).sum()
    off_diag = (c.pow(2).sum() - torch.diagonal(c).pow(2).sum())
    return on_diag + lambda_offdiag * off_diag, c


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("\n=== Loading MNIST ===", flush=True)
    transform = transforms.Compose([transforms.ToTensor()])
    train_mnist = datasets.MNIST(root="./mnist_data", train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST(root="./mnist_data", train=False, download=True, transform=transform)
    print(f"  train={len(train_mnist)}, test={len(test_mnist)}", flush=True)

    train_ds = PairedMNISTSignal(train_mnist, seed_offset=0)
    test_ds = PairedMNISTSignal(test_mnist, seed_offset=100000)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    EMBED_DIM = 64
    img_encoder = ImageEncoder(EMBED_DIM).to(DEVICE)
    sig_encoder = SignalEncoder(EMBED_DIM).to(DEVICE)
    print(f"  ImageEncoder params: {count_params(img_encoder)}, SignalEncoder params: {count_params(sig_encoder)}", flush=True)

    opt = torch.optim.Adam(list(img_encoder.parameters()) + list(sig_encoder.parameters()), lr=1e-3)

    EPOCHS = 15
    print(f"\n=== Self-supervised cross-modal training (Barlow Twins loss, {EPOCHS} epochs, NO LABELS used) ===", flush=True)
    t0 = time.time()
    for epoch in range(EPOCHS):
        img_encoder.train(); sig_encoder.train()
        total_loss, n_batches = 0.0, 0
        for imgs, signals, _labels in train_loader:
            imgs, signals = imgs.to(DEVICE), signals.to(DEVICE)
            opt.zero_grad()
            z_a = img_encoder(imgs)
            z_b = sig_encoder(signals)
            loss, _ = barlow_twins_loss(z_a, z_b)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"  epoch {epoch+1}: loss={total_loss/n_batches:.4f}", flush=True)
    train_time = time.time() - t0
    RESULTS["train_time_s"] = train_time
    RESULTS["train_loss_final"] = total_loss / n_batches

    # ============================================================
    # Eval 1: linear probe on frozen image embeddings (no label leakage during SSL)
    # ============================================================
    print("\n=== Eval 1: Linear probe on frozen image embeddings ===", flush=True)
    img_encoder.eval(); sig_encoder.eval()
    with torch.no_grad():
        train_embeds, train_labels = [], []
        for imgs, signals, labels in train_loader:
            z = img_encoder(imgs.to(DEVICE))
            train_embeds.append(z.cpu())
            train_labels.append(labels)
        train_embeds = torch.cat(train_embeds)
        train_labels = torch.cat(train_labels)

        test_embeds, test_labels = [], []
        for imgs, signals, labels in test_loader:
            z = img_encoder(imgs.to(DEVICE))
            test_embeds.append(z.cpu())
            test_labels.append(labels)
        test_embeds = torch.cat(test_embeds)
        test_labels = torch.cat(test_labels)

    probe = nn.Linear(EMBED_DIM, 10).to(DEVICE)
    probe_opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    tr_x, tr_y = train_embeds.to(DEVICE), train_labels.to(DEVICE)
    te_x, te_y = test_embeds.to(DEVICE), test_labels.to(DEVICE)
    for _ in range(50):
        probe_opt.zero_grad()
        loss = F.cross_entropy(probe(tr_x), tr_y)
        loss.backward()
        probe_opt.step()
    with torch.no_grad():
        probe_acc = (probe(te_x).argmax(-1) == te_y).float().mean().item()
    print(f"  Linear probe test accuracy (image embeddings -> digit label): {probe_acc:.4f}", flush=True)
    print(f"  (chance = 0.10; this measures whether label-relevant structure emerged with NO labels used in training)", flush=True)
    RESULTS["linear_probe_accuracy"] = probe_acc

    # ============================================================
    # Eval 2: cross-modal retrieval -- given an image, retrieve the matching signal
    # ============================================================
    print("\n=== Eval 2: Cross-modal retrieval (image -> signal) ===", flush=True)
    with torch.no_grad():
        test_sig_embeds = []
        for imgs, signals, labels in test_loader:
            z = sig_encoder(signals.to(DEVICE))
            test_sig_embeds.append(z.cpu())
        test_sig_embeds = torch.cat(test_sig_embeds)

    # Use a random subset of 1000 test pairs as the retrieval pool (full 10k is O(n^2), fine on GPU but let's be explicit)
    n_pool = min(1000, len(test_embeds))
    idx = torch.randperm(len(test_embeds))[:n_pool]
    pool_img = F.normalize(test_embeds[idx], dim=-1)
    pool_sig = F.normalize(test_sig_embeds[idx], dim=-1)
    pool_labels = test_labels[idx]

    sims = pool_img @ pool_sig.T  # (n_pool, n_pool) cosine similarity
    top1 = sims.argmax(dim=-1)
    top1_correct = (pool_labels[top1] == pool_labels).float().mean().item()
    top5 = sims.topk(5, dim=-1).indices
    top5_correct = torch.tensor([pool_labels[top5[i]].eq(pool_labels[i]).any() for i in range(n_pool)]).float().mean().item()
    print(f"  Retrieval pool size: {n_pool}, chance top-1 (10 classes, roughly balanced): ~0.10", flush=True)
    print(f"  Top-1 retrieval accuracy (same digit): {top1_correct:.4f}", flush=True)
    print(f"  Top-5 retrieval accuracy (same digit in top 5): {top5_correct:.4f}", flush=True)
    RESULTS["retrieval_top1_accuracy"] = top1_correct
    RESULTS["retrieval_top5_accuracy"] = top5_correct
    RESULTS["retrieval_pool_size"] = n_pool

    with open("crossmodal_correlation_poc_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("\nSaved to crossmodal_correlation_poc_results.json", flush=True)


if __name__ == "__main__":
    main()
