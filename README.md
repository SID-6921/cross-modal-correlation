# Cross-Modal Correlation Alignment

Code and results for a proof-of-concept study testing whether a Barlow-Twins-style
cross-correlation objective — originally designed for two augmented views of a single
modality — works when adapted to align two genuinely different modalities (image ↔
audio, image ↔ text), without any negative pairs or contrastive loss.

**Paper:** [arXiv link to be added once submitted]

## Summary of findings

Two real, independently reported findings:

1. **At closed-set scale** (MNIST paired with a synthetic signal, and MNIST paired
   with real spoken-digit audio), the objective works — near-ceiling accuracy and
   retrieval, confirmed by negative controls (mismatched pairs collapse to chance),
   5-seed replication, and a real InfoNCE contrastive baseline the method matches or
   beats on retrieval.
2. **At open-vocabulary scale** (real photographs paired with real free-form captions,
   Flickr8k), the same objective loses decisively to InfoNCE — by 4–7x on retrieval,
   stably across 3 seeds and confirmed by its own negative control.

Both results are reported with matched rigor. The open-vocabulary result is treated as
a central finding, not a caveat — see the paper for the full discussion, including an
explicitly acknowledged confound (cardinality vs. open-vocabulary vs. frozen backbone)
that this study does not disentangle.

## Repository contents

`poc/` — all experiment scripts and their real result files (JSON/JSONL), one pair per
experiment reported in the paper:

| Script | What it runs |
|---|---|
| `run_crossmodal_correlation_poc.py` | Experiment 1: MNIST + synthetic per-class signal |
| `run_crossmodal_correlation_poc_hard.py` | Experiment 2: MNIST + real spoken-digit audio (FSDD) |
| `run_negative_control_easy.py` / `run_negative_control_hard.py` | Negative controls (mismatched pairs) for Experiments 1 and 2 |
| `run_contrastive_baseline_hard.py` | InfoNCE contrastive baseline on Experiment 2's data/architecture |
| `run_multiseed_hard.py` / `run_multiseed_hard_infonce.py` | 5-seed replication, Barlow Twins and InfoNCE respectively |
| `run_interpretability_check.py` | Inspects the learned cross-correlation matrix directly |
| `run_openvocab_flickr8k.py` | Open-vocabulary experiment: Flickr8k images + captions, both objectives |
| `run_openvocab_negative_control.py` | Negative control for the Flickr8k experiment |

All `*_results.json` / `*_results.jsonl` files are real outputs from real training runs —
nothing in this repository is illustrative or simulated.

## Reproducing

Each script is self-contained (PyTorch + torchvision; `run_crossmodal_correlation_poc_hard.py`
and related scripts additionally need `soundfile` and the Free Spoken Digit Dataset;
`run_openvocab_*.py` scripts need the Flickr8k dataset and PIL). Datasets are not included
in this repository — MNIST downloads automatically via `torchvision.datasets.MNIST`; FSDD
and Flickr8k are linked below.

- MNIST: via `torchvision.datasets.MNIST` (automatic download)
- Free Spoken Digit Dataset: https://github.com/Jakobovski/free-spoken-digit-dataset
- Flickr8k: standard public image-captioning dataset (Hodosh, Young & Hockenmaier, 2013)

## License

No license file included yet — treat as all-rights-reserved until one is added.
