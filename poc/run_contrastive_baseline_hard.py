"""
Contrastive (InfoNCE) baseline on the SAME real-data task as Experiment 2
(MNIST + real FSDD speech), addressing the review's MAJOR finding: no
comparison point exists against the dominant contrastive family (CLIP-
style) this paper implicitly positions itself against.

Same encoders, same data, same evaluation protocol as the Barlow-Twins-
style experiment -- the ONLY difference is the training objective:
standard symmetric InfoNCE contrastive loss (as in CLIP) instead of the
cross-correlation objective. This isolates the effect of the objective
itself, holding everything else fixed.
"""
import json
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

RESULTS = {"device": str(DEVICE), "experiment": "contrastive_infonce_baseline_hard"}

AUDIO_LEN = 4000
FSDD_DIR = "free-spoken-digit-dataset-master/recordings"


def load_fsdd():
    files = sorted(os.listdir(FSDD_DIR))
    by_digit = {d: [] for d in range(10)}
    for fname in files:
        digit = int(fname.split("_")[0])
        audio, sr = sf.read(os.path.join(FSDD_DIR, fname))
        if len(audio) >= AUDIO_LEN:
            audio = audio[:AUDIO_LEN]
        else:
            audio = np.pad(audio, (0, AUDIO_LEN - len(audio)))
        audio = audio / (np.abs(audio).max() + 1e-8)
        by_digit[digit].append(torch.tensor(audio, dtype=torch.float32))
    return by_digit


class PairedMNISTSpeech(torch.utils.data.Dataset):
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
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, embed_dim),
        )
    def forward(self, x): return self.net(x)


class AudioEncoder(nn.Module):
    def __init__(self, embed_dim=64, audio_len=AUDIO_LEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 80, stride=4, padding=38), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
            nn.Flatten(), nn.Linear(64 * 8, 128), nn.ReLU(), nn.Linear(128, embed_dim),
        )
    def forward(self, x): return self.net(x.unsqueeze(1))


def info_nce_loss(z_a, z_b, temperature=0.1):
    """Standard symmetric InfoNCE / CLIP-style contrastive loss: each
    matched pair (i,i) in the batch is the positive; all other in-batch
    pairs (i,j), i != j are negatives."""
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = z_a @ z_b.T / temperature  # (n, n)
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_b) / 2


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("\n=== Loading real datasets: MNIST + FSDD (contrastive baseline) ===", flush=True)
    transform = transforms.Compose([transforms.ToTensor()])
    train_mnist = datasets.MNIST(root="./mnist_data", train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST(root="./mnist_data", train=False, download=True, transform=transform)
    audio_by_digit = load_fsdd()

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
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    EMBED_DIM = 64
    img_encoder = ImageEncoder(EMBED_DIM).to(DEVICE)
    aud_encoder = AudioEncoder(EMBED_DIM).to(DEVICE)
    print(f"  ImageEncoder params: {count_params(img_encoder)}, AudioEncoder params: {count_params(aud_encoder)}", flush=True)

    opt = torch.optim.Adam(list(img_encoder.parameters()) + list(aud_encoder.parameters()), lr=1e-3)

    EPOCHS = 20
    print(f"\n=== Contrastive (InfoNCE) training, same data as real experiment, {EPOCHS} epochs ===", flush=True)
    t0 = time.time()
    for epoch in range(EPOCHS):
        img_encoder.train(); aud_encoder.train()
        total_loss, n_batches = 0.0, 0
        for imgs, audios, _labels in train_loader:
            imgs, audios = imgs.to(DEVICE), audios.to(DEVICE)
            opt.zero_grad()
            z_a = img_encoder(imgs)
            z_b = aud_encoder(audios)
            loss = info_nce_loss(z_a, z_b)
            loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1
        print(f"  epoch {epoch+1}: loss={total_loss/n_batches:.4f}", flush=True)
    RESULTS["train_time_s"] = time.time() - t0
    RESULTS["train_loss_final"] = total_loss / n_batches

    print("\n=== Eval 1: Linear probe on frozen image embeddings ===", flush=True)
    img_encoder.eval(); aud_encoder.eval()
    with torch.no_grad():
        train_embeds, train_labels = [], []
        for imgs, audios, labels in train_loader:
            z = img_encoder(imgs.to(DEVICE))
            train_embeds.append(z.cpu()); train_labels.append(labels)
        train_embeds = torch.cat(train_embeds); train_labels = torch.cat(train_labels)

        test_embeds, test_labels, test_aud_embeds = [], [], []
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
        loss.backward(); probe_opt.step()
    with torch.no_grad():
        probe_acc = (probe(te_x).argmax(-1) == te_y).float().mean().item()
    print(f"  Linear probe test accuracy (InfoNCE baseline): {probe_acc:.4f}", flush=True)
    RESULTS["linear_probe_accuracy"] = probe_acc

    print("\n=== Eval 2: Cross-modal retrieval (InfoNCE baseline) ===", flush=True)
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
    print(f"  Top-1 retrieval accuracy (InfoNCE baseline): {top1_correct:.4f}", flush=True)
    print(f"  Top-5 retrieval accuracy (InfoNCE baseline): {top5_correct:.4f}", flush=True)
    RESULTS["retrieval_top1_accuracy"] = top1_correct
    RESULTS["retrieval_top5_accuracy"] = top5_correct
    RESULTS["retrieval_pool_size"] = n_pool

    print("\n=== Comparison to Barlow-Twins-style result (Experiment 2) ===", flush=True)
    print("  Barlow Twins: linear_probe=0.9920, retrieval_top1=0.9910", flush=True)
    print(f"  InfoNCE:      linear_probe={probe_acc:.4f}, retrieval_top1={top1_correct:.4f}", flush=True)

    with open("contrastive_baseline_hard_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("\nSaved to contrastive_baseline_hard_results.json", flush=True)


if __name__ == "__main__":
    main()
