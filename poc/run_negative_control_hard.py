"""
Negative control for Experiment 2 (MNIST + real speech, FSDD): same
mismatched-pairing test as run_negative_control_easy.py, but on the
harder, real-data experiment.
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

RESULTS = {"device": str(DEVICE), "experiment": "negative_control_hard_random_mismatched_pairs"}

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


class MismatchedMNISTSpeech(torch.utils.data.Dataset):
    """NEGATIVE CONTROL: pairs each image with a real recording of a
    RANDOM DIFFERENT digit (never the true digit)."""
    def __init__(self, mnist_dataset, audio_by_digit, seed=0):
        self.mnist = mnist_dataset
        self.audio_by_digit = audio_by_digit
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, true_label = self.mnist[idx]
        wrong_digit = int(self.rng.integers(0, 10))
        while wrong_digit == true_label:
            wrong_digit = int(self.rng.integers(0, 10))
        candidates = self.audio_by_digit[wrong_digit]
        audio = candidates[self.rng.integers(0, len(candidates))]
        return img, audio, true_label


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


def barlow_twins_loss(z_a, z_b, lambda_offdiag=0.005):
    n, d = z_a.shape
    z_a_norm = (z_a - z_a.mean(0)) / (z_a.std(0) + 1e-8)
    z_b_norm = (z_b - z_b.mean(0)) / (z_b.std(0) + 1e-8)
    c = (z_a_norm.T @ z_b_norm) / n
    on_diag = ((torch.diagonal(c) - 1) ** 2).sum()
    off_diag = (c.pow(2).sum() - torch.diagonal(c).pow(2).sum())
    return on_diag + lambda_offdiag * off_diag


def main():
    print("\n=== Loading real datasets (negative control: MISMATCHED pairs) ===", flush=True)
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

    train_ds = MismatchedMNISTSpeech(train_mnist, train_audio, seed=0)
    test_ds = MismatchedMNISTSpeech(test_mnist, test_audio, seed=1000)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    EMBED_DIM = 64
    img_encoder = ImageEncoder(EMBED_DIM).to(DEVICE)
    aud_encoder = AudioEncoder(EMBED_DIM).to(DEVICE)
    opt = torch.optim.Adam(list(img_encoder.parameters()) + list(aud_encoder.parameters()), lr=1e-3)

    EPOCHS = 20
    print(f"\n=== Training on MISMATCHED real pairs, {EPOCHS} epochs ===", flush=True)
    t0 = time.time()
    for epoch in range(EPOCHS):
        img_encoder.train(); aud_encoder.train()
        total_loss, n_batches = 0.0, 0
        for imgs, audios, _true_labels in train_loader:
            imgs, audios = imgs.to(DEVICE), audios.to(DEVICE)
            opt.zero_grad()
            z_a = img_encoder(imgs)
            z_b = aud_encoder(audios)
            loss = barlow_twins_loss(z_a, z_b)
            loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1
        print(f"  epoch {epoch+1}: loss={total_loss/n_batches:.4f}", flush=True)
    RESULTS["train_time_s"] = time.time() - t0
    RESULTS["train_loss_final"] = total_loss / n_batches

    print("\n=== Eval: Linear probe (TRUE labels, trained on MISMATCHED real pairs) ===", flush=True)
    img_encoder.eval(); aud_encoder.eval()
    with torch.no_grad():
        train_embeds, train_labels = [], []
        for imgs, audios, labels in train_loader:
            z = img_encoder(imgs.to(DEVICE))
            train_embeds.append(z.cpu()); train_labels.append(labels)
        train_embeds = torch.cat(train_embeds); train_labels = torch.cat(train_labels)

        test_embeds, test_labels = [], []
        for imgs, audios, labels in test_loader:
            z = img_encoder(imgs.to(DEVICE))
            test_embeds.append(z.cpu()); test_labels.append(labels)
        test_embeds = torch.cat(test_embeds); test_labels = torch.cat(test_labels)

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
    print(f"  Linear probe test accuracy under MISMATCHED-pair training (real speech): {probe_acc:.4f}", flush=True)
    print(f"  (Real experiment's matched-pair result was 0.9920. Chance = 0.10.)", flush=True)
    RESULTS["linear_probe_accuracy_mismatched"] = probe_acc
    RESULTS["linear_probe_accuracy_real_experiment_for_comparison"] = 0.9920

    with open("negative_control_hard_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("\nSaved to negative_control_hard_results.json", flush=True)


if __name__ == "__main__":
    main()
