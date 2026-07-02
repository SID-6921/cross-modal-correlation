"""
Negative control for Experiment 1 (MNIST + synthetic signal), addressing
the Devil's Advocate's CRITICAL finding: does linear probe accuracy stay
high even with RANDOMLY MISMATCHED image-signal pairs during training?

If yes: the 98.89% linear probe result in the real paper says nothing
about cross-modal alignment -- it just reflects the image encoder
learning to classify MNIST via some path independent of the audio/signal
side, since MNIST classification is trivially easy for any reasonable
CNN regardless of the training objective's cross-modal correctness.

If no (accuracy collapses toward chance): this validates that the real
experiment's high accuracy genuinely depends on correct cross-modal
pairing, not an artifact of task ease.

Identical architecture/hyperparameters to the real experiment, the ONLY
change is: training pairs are randomly mismatched (image of digit k
paired with signal generated for a DIFFERENT random digit), while
evaluation still measures probe accuracy against the TRUE image label.
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

RESULTS = {"device": str(DEVICE), "experiment": "negative_control_easy_random_mismatched_pairs"}

SIGNAL_LEN = 64
BASE_FREQS = [3 + 2 * d for d in range(10)]


def make_signal_for_digit(digit, seed_offset=0):
    g = torch.Generator().manual_seed(hash((digit, seed_offset)) % (2**31))
    t = torch.arange(SIGNAL_LEN).float()
    freq = BASE_FREQS[digit]
    phase = torch.rand(1, generator=g).item() * 2 * math.pi
    noise = torch.randn(SIGNAL_LEN, generator=g) * 0.15
    return torch.sin(2 * math.pi * freq * t / SIGNAL_LEN + phase) + noise


class MismatchedMNISTSignal(torch.utils.data.Dataset):
    """NEGATIVE CONTROL: pairs each image with a signal for a RANDOM
    DIFFERENT digit, not its own -- breaks the true cross-modal
    correspondence while keeping everything else identical."""
    def __init__(self, mnist_dataset, seed_offset=0):
        self.mnist = mnist_dataset
        self.seed_offset = seed_offset
        g = torch.Generator().manual_seed(seed_offset)
        # Fixed random wrong-digit assignment per sample (not the true label)
        self.fake_digits = torch.randint(0, 10, (len(mnist_dataset),), generator=g)
        # Ensure the fake digit never equals the true digit (genuine mismatch)
        for i in range(len(mnist_dataset)):
            true_label = mnist_dataset[i][1]
            while self.fake_digits[i].item() == true_label:
                self.fake_digits[i] = torch.randint(0, 10, (1,), generator=g).item()

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, true_label = self.mnist[idx]
        fake_digit = self.fake_digits[idx].item()
        signal = make_signal_for_digit(fake_digit, seed_offset=self.seed_offset + idx)
        return img, signal, true_label  # eval still uses TRUE label


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, embed_dim),
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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("\n=== Loading MNIST (negative control: RANDOMLY MISMATCHED pairs) ===", flush=True)
    transform = transforms.Compose([transforms.ToTensor()])
    train_mnist = datasets.MNIST(root="./mnist_data", train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST(root="./mnist_data", train=False, download=True, transform=transform)

    train_ds = MismatchedMNISTSignal(train_mnist, seed_offset=0)
    test_ds = MismatchedMNISTSignal(test_mnist, seed_offset=100000)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    EMBED_DIM = 64
    img_encoder = ImageEncoder(EMBED_DIM).to(DEVICE)
    sig_encoder = SignalEncoder(EMBED_DIM).to(DEVICE)
    opt = torch.optim.Adam(list(img_encoder.parameters()) + list(sig_encoder.parameters()), lr=1e-3)

    EPOCHS = 15
    print(f"\n=== Training on MISMATCHED pairs, {EPOCHS} epochs ===", flush=True)
    t0 = time.time()
    for epoch in range(EPOCHS):
        img_encoder.train(); sig_encoder.train()
        total_loss, n_batches = 0.0, 0
        for imgs, signals, _true_labels in train_loader:
            imgs, signals = imgs.to(DEVICE), signals.to(DEVICE)
            opt.zero_grad()
            z_a = img_encoder(imgs)
            z_b = sig_encoder(signals)
            loss = barlow_twins_loss(z_a, z_b)
            loss.backward()
            opt.step()
            total_loss += loss.item(); n_batches += 1
        print(f"  epoch {epoch+1}: loss={total_loss/n_batches:.4f}", flush=True)
    RESULTS["train_time_s"] = time.time() - t0
    RESULTS["train_loss_final"] = total_loss / n_batches

    print("\n=== Eval: Linear probe on frozen image embeddings (TRUE labels, trained on MISMATCHED pairs) ===", flush=True)
    img_encoder.eval(); sig_encoder.eval()
    with torch.no_grad():
        train_embeds, train_labels = [], []
        for imgs, signals, labels in train_loader:
            z = img_encoder(imgs.to(DEVICE))
            train_embeds.append(z.cpu()); train_labels.append(labels)
        train_embeds = torch.cat(train_embeds); train_labels = torch.cat(train_labels)

        test_embeds, test_labels = [], []
        for imgs, signals, labels in test_loader:
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
    print(f"  Linear probe test accuracy under MISMATCHED-pair training: {probe_acc:.4f}", flush=True)
    print(f"  (Real experiment's matched-pair result was 0.9889. Chance = 0.10.)", flush=True)
    print(f"  If this stays high (~0.98+), the real result is likely an artifact of MNIST being trivially classifiable,", flush=True)
    print(f"  NOT evidence of genuine cross-modal alignment. If this collapses toward chance, alignment is real.", flush=True)
    RESULTS["linear_probe_accuracy_mismatched"] = probe_acc
    RESULTS["linear_probe_accuracy_real_experiment_for_comparison"] = 0.9889

    with open("negative_control_easy_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("\nSaved to negative_control_easy_results.json", flush=True)


if __name__ == "__main__":
    main()
