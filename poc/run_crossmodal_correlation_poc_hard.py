"""
Harder proof-of-concept, following up on the too-easy first POC (which used
a deterministic sine-wave code per digit -- essentially a lookup table).

This version uses REAL data on both sides:
  Modality A: real MNIST digit images.
  Modality B: real human speech recordings of spoken digits (Free Spoken
    Digit Dataset -- Jakobovski/free-spoken-digit-dataset, 3000 recordings,
    multiple speakers, real recording noise, real speaking-rate variability).

Pairing is class-level and many-to-many: any image of digit 3 is validly
paired with any recording of "three" -- there is no 1:1 deterministic code
between a specific image and a specific recording, unlike the first POC.
This is a genuinely harder and more realistic test of whether cross-modal
correlation alignment (Barlow Twins style, no labels used in the loss)
still works when the cross-modal relationship is class-level and noisy,
not a clean bijection.

Every number below is real, from real data, real training.
"""
import json
import math
import os
import time
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

RESULTS = {"device": str(DEVICE)}

AUDIO_LEN = 4000  # fixed length, truncate/pad real recordings (8kHz sample rate -> 0.5s)
FSDD_DIR = "free-spoken-digit-dataset-master/recordings"


def load_fsdd():
    """Parse FSDD filenames ({digit}_{speaker}_{index}.wav), load and
    fixed-length-normalize each real recording."""
    files = sorted(os.listdir(FSDD_DIR))
    by_digit = {d: [] for d in range(10)}
    for fname in files:
        digit = int(fname.split("_")[0])
        audio, sr = sf.read(os.path.join(FSDD_DIR, fname))
        if len(audio) >= AUDIO_LEN:
            audio = audio[:AUDIO_LEN]
        else:
            audio = np.pad(audio, (0, AUDIO_LEN - len(audio)))
        # normalize amplitude per-clip (real recordings vary in loudness)
        audio = audio / (np.abs(audio).max() + 1e-8)
        by_digit[digit].append(torch.tensor(audio, dtype=torch.float32))
    return by_digit


class PairedMNISTSpeech(torch.utils.data.Dataset):
    """Many-to-many class-level pairing: for each image, sample a RANDOM
    real recording of the same digit (not a fixed one-to-one mapping)."""
    def __init__(self, mnist_dataset, audio_by_digit, seed=0):
        self.mnist = mnist_dataset
        self.audio_by_digit = audio_by_digit
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        candidates = self.audio_by_digit[label]
        audio = candidates[self.rng.integers(0, len(candidates))]
        return img, audio, label


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class AudioEncoder(nn.Module):
    """Real raw-waveform encoder -- deeper than the synthetic-signal
    encoder in the first POC, since real speech has more structure to
    extract (formants, onsets, real noise) than a pure sine tone."""
    def __init__(self, embed_dim=64, audio_len=AUDIO_LEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 80, stride=4, padding=38), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(64 * 8, 128), nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))


def barlow_twins_loss(z_a, z_b, lambda_offdiag=0.005):
    n, d = z_a.shape
    z_a_norm = (z_a - z_a.mean(0)) / (z_a.std(0) + 1e-8)
    z_b_norm = (z_b - z_b.mean(0)) / (z_b.std(0) + 1e-8)
    c = (z_a_norm.T @ z_b_norm) / n
    on_diag = ((torch.diagonal(c) - 1) ** 2).sum()
    off_diag = (c.pow(2).sum() - torch.diagonal(c).pow(2).sum())
    return on_diag + lambda_offdiag * off_diag


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("\n=== Loading real datasets: MNIST + Free Spoken Digit Dataset ===", flush=True)
    transform = transforms.Compose([transforms.ToTensor()])
    train_mnist = datasets.MNIST(root="./mnist_data", train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST(root="./mnist_data", train=False, download=True, transform=transform)

    audio_by_digit = load_fsdd()
    for d in range(10):
        print(f"  digit {d}: {len(audio_by_digit[d])} real recordings", flush=True)

    # Split FSDD recordings into train/test pools per digit (80/20) so test
    # pairing uses genuinely held-out recordings, not just held-out images.
    train_audio, test_audio = {}, {}
    rng = np.random.default_rng(42)
    for d in range(10):
        clips = audio_by_digit[d]
        idx = rng.permutation(len(clips))
        n_test = max(1, int(0.2 * len(clips)))
        test_audio[d] = [clips[i] for i in idx[:n_test]]
        train_audio[d] = [clips[i] for i in idx[n_test:]]

    train_ds = PairedMNISTSpeech(train_mnist, train_audio, seed=0)
    test_ds = PairedMNISTSpeech(test_mnist, test_audio, seed=1000)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    EMBED_DIM = 64
    img_encoder = ImageEncoder(EMBED_DIM).to(DEVICE)
    aud_encoder = AudioEncoder(EMBED_DIM).to(DEVICE)
    print(f"  ImageEncoder params: {count_params(img_encoder)}, AudioEncoder params: {count_params(aud_encoder)}", flush=True)

    opt = torch.optim.Adam(list(img_encoder.parameters()) + list(aud_encoder.parameters()), lr=1e-3)

    EPOCHS = 20
    print(f"\n=== Self-supervised cross-modal training (real MNIST + real speech, {EPOCHS} epochs, NO LABELS used) ===", flush=True)
    t0 = time.time()
    for epoch in range(EPOCHS):
        img_encoder.train(); aud_encoder.train()
        total_loss, n_batches = 0.0, 0
        for imgs, audios, _labels in train_loader:
            imgs, audios = imgs.to(DEVICE), audios.to(DEVICE)
            opt.zero_grad()
            z_a = img_encoder(imgs)
            z_b = aud_encoder(audios)
            loss = barlow_twins_loss(z_a, z_b)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"  epoch {epoch+1}: loss={total_loss/n_batches:.4f}", flush=True)
    train_time = time.time() - t0
    RESULTS["train_time_s"] = train_time
    RESULTS["train_loss_final"] = total_loss / n_batches

    print("\n=== Eval 1: Linear probe on frozen image embeddings ===", flush=True)
    img_encoder.eval(); aud_encoder.eval()
    with torch.no_grad():
        train_embeds, train_labels = [], []
        for imgs, audios, labels in train_loader:
            z = img_encoder(imgs.to(DEVICE))
            train_embeds.append(z.cpu()); train_labels.append(labels)
        train_embeds = torch.cat(train_embeds); train_labels = torch.cat(train_labels)

        test_embeds, test_labels = [], []
        test_aud_embeds = []
        for imgs, audios, labels in test_loader:
            z = img_encoder(imgs.to(DEVICE))
            za = aud_encoder(audios.to(DEVICE))
            test_embeds.append(z.cpu()); test_aud_embeds.append(za.cpu()); test_labels.append(labels)
        test_embeds = torch.cat(test_embeds); test_aud_embeds = torch.cat(test_aud_embeds); test_labels = torch.cat(test_labels)

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
    RESULTS["linear_probe_accuracy"] = probe_acc

    print("\n=== Eval 2: Cross-modal retrieval (image -> real speech recording) ===", flush=True)
    n_pool = min(1000, len(test_embeds))
    idx = torch.randperm(len(test_embeds))[:n_pool]
    pool_img = F.normalize(test_embeds[idx], dim=-1)
    pool_aud = F.normalize(test_aud_embeds[idx], dim=-1)
    pool_labels = test_labels[idx]

    sims = pool_img @ pool_aud.T
    top1 = sims.argmax(dim=-1)
    top1_correct = (pool_labels[top1] == pool_labels).float().mean().item()
    top5 = sims.topk(5, dim=-1).indices
    top5_correct = torch.tensor([pool_labels[top5[i]].eq(pool_labels[i]).any() for i in range(n_pool)]).float().mean().item()
    print(f"  Retrieval pool size: {n_pool}", flush=True)
    print(f"  Top-1 retrieval accuracy (same digit, real held-out speech): {top1_correct:.4f}", flush=True)
    print(f"  Top-5 retrieval accuracy: {top5_correct:.4f}", flush=True)
    RESULTS["retrieval_top1_accuracy"] = top1_correct
    RESULTS["retrieval_top5_accuracy"] = top5_correct
    RESULTS["retrieval_pool_size"] = n_pool

    with open("crossmodal_correlation_poc_hard_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("\nSaved to crossmodal_correlation_poc_hard_results.json", flush=True)


if __name__ == "__main__":
    main()
