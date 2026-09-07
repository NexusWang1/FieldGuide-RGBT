# FieldGuide: Plume-Shift Fire-Source Field Guidance for Smoke Retrieval in UAV RGB-Thermal Early Fire Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22646595.svg)](https://doi.org/10.5281/zenodo.22646595)

Official code for the IEEE GRSL letter:

> **Plume-Shift Fire-Source Field Guidance for Smoke Retrieval in UAV RGB-Thermal Early Fire Detection**
> Wenchen Wang\*, Chengjie Gu\*, Dongjun Zhu†, Huan Liao, and Jingjing Chen — Anhui University of Science and Technology
> (\* equal contribution; † corresponding author)

FieldGuide retrieves small-object smoke that visible-range detectors lose under dense smoke:
a learnable **fire-source field** built from the IR stream is cross-correlated with a learnable
**plume-shift kernel** K (7×7) to produce a spatial **guidance map** S over candidate smoke
regions, which modulates the fused P2 feature at fixed strength α=0.5 while an auxiliary BCE
loss supervises S from the first training step.

## Key findings

- On the official split of RGBT-3M (fp32), FieldGuide improves small-object smoke
  mAP@0.5:0.95 from 14.83% to 15.08% over an identical-architecture two-stream YOLO26
  baseline (two independent runs per arm; noise floor 0.2 pt), with the gain reproduced
  in FLAME2 transfer under a same-protocol ablation.
- A controlled comparison shows that the widely used **zero-initialized multiplicative
  gate keeps the physics-guided pathway permanently inactive** (guidance gate frozen at
  −0.009 after 200 epochs; the fusion gate converges to negative values in six independent
  runs), whereas fixed-strength direct injection activates the mechanism.
- The guidance map aligns with annotated smoke regions 5.81× above background
  (all 1,638 smoke–fire co-occurrence frames of the test split).

## Repository layout

This repo is an **overlay** on top of Ultralytics (tested with Ultralytics 8.4.108,
Python 3.10, PyTorch 2.13 + cu126):

```
overlay/ultralytics/          # copy these files into your ultralytics source tree
  nn/modules/phy_bridge.py            # CMEE / DCMA+BGMA / CPCF / DMFP / PIIP / PSCMT / BMGA
  nn/modules/thermal_field_v1_14.py   # HeatField (fire-source field t) + TFAMBoost
  nn/modules/thermal_field.py         # base thermal-field module (eval plumbing)
  nn/modules/gaussian_kernel.py       # GDK / fixed-Gaussian conv wrappers (KMODE)
  nn/modules/field_guide_v2_3.py      # ★ FieldGuide (gated / fixed modes)
  nn/modules/cbam.py                  # CBAM comparison arm
  nn/tasks_rgbt.py                    # base RGBT model (eval plumbing)
  nn/tasks_rgbt_v1_19.py              # champion backbone base (stage/P2/dual-STAL)
  nn/tasks_rgbt_v2_0.py               # + Gaussian-kernel arm
  nn/tasks_rgbt_v2_3.py               # ★ + FieldGuide injection
  nn/tasks_rgbt_v2_cbam.py            # CBAM neck comparison arm
  models/yolo/detect/train_rgbt.py    # base RGBT trainer (dataset plumbing for eval)
  models/yolo/detect/train_rgbt_v1_19.py
  models/yolo/detect/train_rgbt_v2_3.py
  models/yolo/detect/train_rgbt_v2_cbam.py
  utils/loss_rgbt_v1_14.py            # FireAnchoredV8Loss (fire-anchored smoke assignment)
  data/rgbt_dataset.py                # paired RGB+IR dataset (6-channel)
scripts/
  train_phy_bridge_v2_3.py      # champion training entry (fixed α=0.5 direct injection)
  train_phy_bridge_v2_cbam.py   # CBAM comparison arm entry
  eval_small_ap.py              # overall / small / large AP evaluation
  eval_small_ap_cofire.py       # smoke subset eval split by same-frame visible flame
  train_flame2ft_v3_fgA*.py     # FLAME2 fine-tuning transfer (same-protocol ablation)
  make_degraded_flame2.py       # modality-degradation robustness protocol
  vis/                          # guidance-map / kernel / comparison visualization
configs/
  rgbt_full.yaml                # data config template (edit paths)
```

## Setup

```bash
# 1. install ultralytics 8.4.108 (source checkout recommended)
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics && git checkout v8.4.108 && pip install -e .

# 2. apply the overlay
cp -r /path/to/FieldGuide-RGBT/overlay/ultralytics/* ultralytics/

# 3. prepare RGBT-3M (official split) and edit configs/rgbt_full.yaml
```

## Training

Champion arm (FieldGuide, fixed-strength direct injection, α=0.5):

```bash
python scripts/train_phy_bridge_v2_3.py   # SGD, 200 epochs, fp32
```

Ablation arms (same architecture, same protocol):

| arm | setting |
|---|---|
| baseline (two-stream YOLO26) | `PHY_FG=0` |
| gated FieldGuide (zero-init gate) | `PHY_FG_MODE=gated` (default) |
| direct injection α=0.5 (ours) | `PHY_FG_MODE=fixed PHY_FG_ALPHA=0.5` |
| CBAM neck comparison | `python scripts/train_phy_bridge_v2_cbam.py` |

Key environment variables (all consumed in `scripts/train_phy_bridge_v2_3.py`):
`PHY_STAGE` (3), `PHY_FG` (1/0), `PHY_FG_MODE` (gated|fixed), `PHY_FG_ALPHA` (0.5),
`PHY_FGAUX` (aux BCE weight 0.02), `PHY_FGWARM` (ramp steps 20000),
`PHY_KMODE` (none|gdk|gfix), `PHY_TFAM_IR` (1), `PHY_DUALSTAL`/`PHY_STALW`,
`PHY_SMOKESTAL`/`PHY_SMOKEW`/`PHY_SMOKEDIL`, `PHY_AUXW` (0.1).

## Evaluation

```bash
python scripts/eval_small_ap.py --weights <run>/weights/best.pt \
    --data configs/rgbt_full.yaml --mode rgbt --split val --device 0
# smoke subset split by same-frame visible flame:
python scripts/eval_small_ap_cofire.py --weights <run>/weights/best.pt ...
```

## Expected results (RGBT-3M official test split, fp32, mAP@0.5:0.95)

| model | overall | small smoke |
|---|---|---|
| stock YOLO26 (RGB only, official 100-ep) | 50.99 | 13.72 |
| stock YOLO26 (IR only, official 100-ep) | 28.73 | 2.71 |
| two-stream + CBAM neck (200 ep) | 64.12 | 14.95 |
| baseline two-stream YOLO26 (200 ep) | 64.11 | 14.83 |
| + FieldGuide, zero-init gate | 64.11 | 14.83 (gate frozen) |
| + FieldGuide, direct α=0.5 (ours) | **64.33** | **15.08** |

Per-class (two-run means): smoke 72.49→72.91, fire 60.45→60.55, person 59.17→59.05,
small smoke 14.82→15.04.

## License

This project builds on Ultralytics and is distributed under the **AGPL-3.0** license,
inherited from Ultralytics. Datasets (RGBT-3M, FLAME2) are owned by their respective
authors; please follow their licenses and citations.
