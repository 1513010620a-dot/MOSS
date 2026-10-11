# MOSS

**Mode-Oriented Structuring and Selection for Diffusion-Based Dataset Distillation**

Code: https://github.com/1513010620a-dot/MOSS

This release contains the MOSS construction and hard-label ResNet-18 evaluation
used for the main results. The two launchers reproduce ImageNette and ImageWoof
at IPC-10 and IPC-50: target construction → 10 candidates per slot → one retained
image per slot → three downstream training runs.

## Installation

Use Linux, Python 3.12 and a CUDA GPU. The source was checked on the supplied
RTX 5090 server with PyTorch 2.8.0+cu128 and torchvision 0.23.0+cu128.

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

An existing working environment can also be used. The pinned dependencies reflect
the supplied server; this package does not contain a second copy of diffusers.

Prepare the ImageNet-pretrained `DiT-XL-2-256x256.pt` checkpoint and
`stabilityai/sd-vae-ft-mse` VAE (or its local directory). The frozen selector uses
torchvision's ImageNet-pretrained ResNet-18; its weights download on first use
unless already cached. These weights are distinct from the downstream evaluator,
which initializes its ResNet-18 from scratch.

## Data

Each dataset root must contain `train/<synset>/*.JPEG` and
`val/<synset>/*.JPEG`. The supplied synset lists determine class order.
The defaults match the original server:

- ImageNette: `/root/autodl-tmp/imagenette2`
- ImageWoof: `/root/autodl-tmp/imagewoof2`
- DiT: `pretrained_models/DiT-XL-2-256x256.pt` (falls back to the existing
  `/root/CaO2-main/pretrained_models/DiT-XL-2-256x256.pt`)
- VAE: the existing `/root/autodl-tmp/DiT-XL-2-256/vae`, when present;
  otherwise `stabilityai/sd-vae-ft-mse`

Datasets, model weights and previously generated images are not bundled.

## Run the main experiments

```bash
bash run_nette.sh
bash run_woof.sh
```

Each command runs both IPC budgets. Pass `10` or `50` to run one budget.
For a different machine, set paths explicitly:

```bash
DATA_ROOT=/data/imagenette2 CKPT=/models/DiT-XL-2-256x256.pt VAE_PATH=/models/vae OUTPUT_ROOT=/data/moss/nette bash run_nette.sh
DATA_ROOT=/data/imagewoof2 CKPT=/models/DiT-XL-2-256x256.pt VAE_PATH=/models/vae OUTPUT_ROOT=/data/moss/woof bash run_woof.sh
```

Optional environment settings: `GPU` (default `0`), `PYTHON` (default `python`),
`OUTPUT_ROOT`, and `STAGE` (`all`, `generate`, `select`, or `eval`).
Use `DRY_RUN=1 bash run_nette.sh` to inspect commands without running an experiment.
Use an absolute `OUTPUT_ROOT` when overriding it. Existing candidates are protected;
choose a fresh output directory for a new construction run. For already constructed
data, `STAGE=eval bash run_nette.sh 10` reruns evaluation.

The defaults preserve the supplied main-table configuration: seed 0, DiT-XL/2
at 256 px, 50 sampling steps, CFG 4.0, mode guidance 0.1, stop timestep 25,
and 10 candidates per slot. IPC-10 uses residual global coverage with local
discriminability; IPC-50 uses local per-slot selection. The sampler keeps the
original MGD³ guidance equation. Experimental guidance variants are not exposed
by this release.

## Evaluation and outputs

`evaluation/` contains the hard-label evaluator copied from the supplied
`CaO2-main`, together with its data loader, ResNet and utility dependencies.
The two launchers call this local evaluator, so a separate CaO2 checkout is not
needed. They use ResNet-18 with instance normalization, CutMix, three training
runs, and the original IPC-dependent epoch schedule (2000/1500 for IPC-10/50
on the two 10-class datasets). No soft-label teacher is used.

Candidates, retained datasets and construction logs are written under
`results/<dataset>/ipc<IPC>/` and `results/<dataset>/logs/` by default.
The evaluator writes its original training logs under `evaluation/results/`;
the launcher also captures console output in the corresponding `ipc*_eval.log`.

| Dataset | IPC-10 | IPC-50 |
| --- | --- | --- |
| ImageNette | 62.9 ± 0.5 | 78.5 ± 0.5 |
| ImageWoof | 39.1 ± 0.3 | 59.9 ± 0.4 |
| ImageNet-100 | 25.8 ± 0.5 | 52.6 ± 0.1 |

These are manuscript results, not newly rerun scores from this packaging step.
The core also retains the ImageNet-100 class list and extraction helper. For that
dataset, use the Python entry points with `--spec imagenet100 --nclass 100` for
generation, `--spec custom --class-file misc/class100.txt` for selection, and
`--spec imagenet-100 --nclass 100` for evaluation. Only the requested two dataset
launchers are included.

## Source layout

- `sample_mode_guidance.py`, `mode_discovery.py`, `latent_features.py`: target
  construction and candidate generation.
- `select_candidates_global_coverage.py`, `select_candidates.py`: the two
  main-table selectors.
- `diffusion/`, `models.py`, `download.py`, `data.py`, `misc/`: required dependencies.
- `evaluation/`: CaO2 hard-label evaluation, restricted to the ResNet branch.
- `run_nette.sh`, `run_woof.sh`: the two experiment launchers.

Baseline generation scripts, random selection, ablation/seed-sweep scripts,
plotting tools, generator fine-tuning and the separate GMM usage manual are omitted.
The original server directory remains unchanged.

## Acknowledgments

The implementation builds on MGD³, Meta's DiT, Hugging Face diffusers, and
CaO²'s hard-label evaluation code. Original source copyright notices are retained.
The DiT license is included under `third_party/DiT-LICENSE.txt`.
