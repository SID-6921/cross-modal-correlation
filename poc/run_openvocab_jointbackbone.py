"""
Jointly-trained-backbone ablation for the Flickr8k open-vocabulary
experiment (Section 4.4), addressing the other half of the "confound we
have not disentangled" limitation: was the frozen ResNet18 backbone
disadvantaging Barlow Twins specifically? Identical setup to
run_openvocab_flickr8k.py, except the ResNet18 backbone is now trainable
(fine-tuned jointly with the projection head) instead of frozen, using a
lower learning rate for the backbone than the head (standard fine-tuning
practice) -- everything else (data, evaluation protocol, both objectives)
held fixed.

Usage: python run_openvocab_jointbackbone.py <objective> <seed>
"""
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

OBJECTIVE = sys.argv[1] if len(sys.argv) > 1 else "barlow"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
assert OBJECTIVE in ("barlow", "infonce")

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[jointbackbone {OBJECTIVE} seed {SEED}] Device: {DEVICE}", flush=True)

IMG_DIR = "flickr8k_images/Flicker8k_Dataset"
TEXT_DIR = "flickr8k_text"


def load_captions():
    caps = {}
    with open(os.path.join(TEXT_DIR, "Flickr8k.token.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, cap = line.split("\t")
            fname = key.split("#")[0]
            caps.setdefault(fname, []).append(cap)
    return caps


def load_split(name):
    with open(os.path.join(TEXT_DIR, name), encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def tokenize(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]", " ", s)
    return s.split()


def build_vocab(captions_by_img, train_files, min_freq=3):
    counter = Counter()
    for fname in train_files:
        for cap in captions_by_img.get(fname, []):
            counter.update(tokenize(cap))
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode_caption(cap, vocab, max_len=25):
    ids = [vocab.get(w, vocab["<unk>"]) for w in tokenize(cap)][:max_len]
    ids = ids + [vocab["<pad>"]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class Flickr8kPairs(torch.utils.data.Dataset):
    def __init__(self, files, captions_by_img, vocab, seed=0):
        self.files = files
        self.captions_by_img = captions_by_img
        self.vocab = vocab
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
        img = IMG_TRANSFORM(img)
        caps = self.captions_by_img[fname]
        cap = caps[self.rng.integers(0, len(caps))]
        cap_ids = encode_caption(cap, self.vocab)
        return img, cap_ids, fname


class ImageEncoder(nn.Module):
    """Same as run_openvocab_flickr8k.py, but the backbone is trainable."""
    def __init__(self, embed_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        # NOT frozen this time -- the ablation.
        self.backbone = backbone
        self.proj = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, embed_dim))

    def forward(self, x):
        feats = self.backbone(x)
        return self.proj(feats)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, emb_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(emb_size, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, embed_dim))

    def forward(self, token_ids):
        emb = self.embedding(token_ids).transpose(1, 2)
        pooled = self.conv(emb).squeeze(-1)
        return self.proj(pooled)


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
    print(f"\n=== [jointbackbone {OBJECTIVE} seed {SEED}] Loading real Flickr8k data ===", flush=True)
    captions_by_img = load_captions()
    train_files = [f for f in load_split("Flickr_8k.trainImages.txt") if f in captions_by_img]
    test_files = [f for f in load_split("Flickr_8k.testImages.txt") if f in captions_by_img]

    vocab = build_vocab(captions_by_img, train_files, min_freq=3)
    print(f"  vocabulary size: {len(vocab)}", flush=True)

    train_ds = Flickr8kPairs(train_files, captions_by_img, vocab, seed=SEED)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, drop_last=True)

    EMBED_DIM = 128
    img_encoder = ImageEncoder(EMBED_DIM).to(DEVICE)
    txt_encoder = TextEncoder(len(vocab), EMBED_DIM).to(DEVICE)
    # Differential learning rate: lower for the pretrained backbone, standard
    # fine-tuning practice, higher for randomly-initialized new params.
    opt = torch.optim.Adam([
        {"params": img_encoder.backbone.parameters(), "lr": 1e-4},
        {"params": img_encoder.proj.parameters(), "lr": 1e-3},
        {"params": txt_encoder.parameters(), "lr": 1e-3},
    ])
    loss_fn = barlow_twins_loss if OBJECTIVE == "barlow" else info_nce_loss

    EPOCHS = 15
    print(f"\n=== [jointbackbone {OBJECTIVE} seed {SEED}] Training, {EPOCHS} epochs, backbone trainable ===", flush=True)
    t0 = time.time()
    for epoch in range(EPOCHS):
        img_encoder.train(); txt_encoder.train()
        total_loss, n_batches = 0.0, 0
        for imgs, caps, _fnames in train_loader:
            imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)
            opt.zero_grad()
            z_img = img_encoder(imgs)
            z_txt = txt_encoder(caps)
            loss = loss_fn(z_img, z_txt)
            loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1
        print(f"  [jointbackbone {OBJECTIVE} seed {SEED}] epoch {epoch+1}: loss={total_loss/n_batches:.4f} "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)
    train_time = time.time() - t0

    print(f"\n=== [jointbackbone {OBJECTIVE} seed {SEED}] Evaluation ===", flush=True)
    img_encoder.eval(); txt_encoder.eval()
    with torch.no_grad():
        img_embeds, img_fnames = [], []
        for fname in test_files:
            img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
            img = IMG_TRANSFORM(img).unsqueeze(0).to(DEVICE)
            img_embeds.append(img_encoder(img).cpu())
            img_fnames.append(fname)
        img_embeds = torch.cat(img_embeds)

        pool_embeds, pool_fnames = [], []
        batch_caps, batch_owners = [], []
        for fname in test_files:
            for cap in captions_by_img[fname]:
                batch_caps.append(encode_caption(cap, vocab))
                batch_owners.append(fname)
        batch_caps_t = torch.stack(batch_caps).to(DEVICE)
        for i in range(0, len(batch_caps_t), 256):
            chunk = batch_caps_t[i:i+256]
            pool_embeds.append(txt_encoder(chunk).cpu())
        pool_embeds = torch.cat(pool_embeds)
        pool_fnames = batch_owners

    img_n = F.normalize(img_embeds, dim=-1)
    txt_n = F.normalize(pool_embeds, dim=-1)
    sims = img_n @ txt_n.T

    def recall_at_k(k):
        topk = sims.topk(k, dim=-1).indices
        hits = 0
        for i, fname in enumerate(img_fnames):
            retrieved_owners = [pool_fnames[j] for j in topk[i].tolist()]
            if fname in retrieved_owners:
                hits += 1
        return hits / len(img_fnames)

    r1, r5, r10 = recall_at_k(1), recall_at_k(5), recall_at_k(10)
    print(f"  [jointbackbone {OBJECTIVE} seed {SEED}] Recall@1={r1:.4f} R@5={r5:.4f} R@10={r10:.4f}", flush=True)

    result = {
        "objective": OBJECTIVE, "seed": SEED, "backbone": "jointly-trained",
        "vocab_size": len(vocab), "retrieval_pool_size": len(pool_fnames),
        "recall_at_1": r1, "recall_at_5": r5, "recall_at_10": r10,
        "train_time_s": train_time, "train_loss_final": total_loss / n_batches,
    }
    out_file = f"openvocab_jointbackbone_{OBJECTIVE}_seed{SEED}_results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
