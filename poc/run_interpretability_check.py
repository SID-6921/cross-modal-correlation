"""
Checks the Discussion section's claim that the cross-correlation matrix is
a directly-inspectable, interpretable object. Trains once (identical setup
to Experiment 2, seed 0) and reports real statistics on the learned
cross-correlation matrix: mean on-diagonal value (should be near 1 if the
claim holds), mean absolute off-diagonal value (should be near 0), and
whether the diagonal is actually distinguishable from the off-diagonal
distribution -- i.e., whether "reading" the matrix would tell you anything
a contrastive loss's embeddings would not directly hand you.
"""
import json
import os
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

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


def barlow_twins_loss_and_matrix(z_a, z_b, lambda_offdiag=0.005):
    n, d = z_a.shape
    z_a_norm = (z_a - z_a.mean(0)) / (z_a.std(0) + 1e-8)
    z_b_norm = (z_b - z_b.mean(0)) / (z_b.std(0) + 1e-8)
    c = (z_a_norm.T @ z_b_norm) / n
    on_diag = ((torch.diagonal(c) - 1) ** 2).sum()
    off_diag = (c.pow(2).sum() - torch.diagonal(c).pow(2).sum())
    return on_diag + lambda_offdiag * off_diag, c


def main():
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
    opt = torch.optim.Adam(list(img_encoder.parameters()) + list(aud_encoder.parameters()), lr=1e-3)

    print("\n=== Training (seed 0, identical to Experiment 2) ===", flush=True)
    for epoch in range(20):
        img_encoder.train(); aud_encoder.train()
        total_loss, n_batches = 0.0, 0
        for imgs, audios, _labels in train_loader:
            imgs, audios = imgs.to(DEVICE), audios.to(DEVICE)
            opt.zero_grad()
            z_a = img_encoder(imgs)
            z_b = aud_encoder(audios)
            loss, _ = barlow_twins_loss_and_matrix(z_a, z_b)
            loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1
        print(f"  epoch {epoch+1}: loss={total_loss/n_batches:.4f}", flush=True)

    print("\n=== Inspecting learned cross-correlation matrix on held-out test batch ===", flush=True)
    img_encoder.eval(); aud_encoder.eval()
    with torch.no_grad():
        imgs, audios, labels = next(iter(test_loader))
        z_a = img_encoder(imgs.to(DEVICE))
        z_b = aud_encoder(audios.to(DEVICE))
        _, c = barlow_twins_loss_and_matrix(z_a, z_b)
        c = c.cpu().numpy()

    diag = np.diagonal(c)
    off_diag_mask = ~np.eye(c.shape[0], dtype=bool)
    off_diag = c[off_diag_mask]

    result = {
        "diag_mean": float(diag.mean()),
        "diag_std": float(diag.std()),
        "diag_min": float(diag.min()),
        "diag_max": float(diag.max()),
        "offdiag_mean_abs": float(np.abs(off_diag).mean()),
        "offdiag_std": float(off_diag.std()),
        "offdiag_max_abs": float(np.abs(off_diag).max()),
        "separation_ratio_diag_over_offdiag": float(diag.mean() / (np.abs(off_diag).mean() + 1e-8)),
    }
    print(f"  Diagonal:     mean={result['diag_mean']:.4f}  std={result['diag_std']:.4f}  "
          f"range=[{result['diag_min']:.4f}, {result['diag_max']:.4f}]", flush=True)
    print(f"  Off-diagonal: mean|.|={result['offdiag_mean_abs']:.4f}  std={result['offdiag_std']:.4f}  "
          f"max|.|={result['offdiag_max_abs']:.4f}", flush=True)
    print(f"  Separation ratio (diag mean / off-diag mean|.|): {result['separation_ratio_diag_over_offdiag']:.2f}x", flush=True)

    with open("interpretability_check_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to interpretability_check_results.json", flush=True)


if __name__ == "__main__":
    main()
