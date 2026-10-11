#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select one image per V3 slot from multiple generated candidate datasets.

Selection rule (no extra tuned weight):
    score(x) = cos(f(x), f(slot_prototype)) - max_{j != c} cos(f(x), class_centroid_j)

Safety rule:
    Prefer candidates whose class-centroid margin is positive:
        cos(f(x), class_centroid_c) > max_{j != c} cos(f(x), class_centroid_j)
    If none of the candidates for a slot satisfies this, fall back to the highest score.

The slot prototype is the real image stored in V3's clusters_centers_path,
i.e. the real sample nearest to that final V3 sub-center.

Expected candidate layout:
    candidate_root/
      dataset_0/<class>/<slot>.png
      dataset_1/<class>/<slot>.png
      ...
      v3_modes.pkl

Example:
    python select_candidates.py \
      --candidate-root /root/autodl-tmp/v3_candidates_woof_ipc10 \
      --real-root /root/autodl-tmp/imagewoof2 \
      --cluster-file /root/autodl-tmp/v3_candidates_woof_ipc10/v3_modes.pkl \
      --spec woof \
      --output /root/autodl-tmp/v3_selected_woof_ipc10 \
      --batch-size 128
"""

import argparse
import csv
import json
import os
import pickle
import re
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def numeric_dataset_key(path: Path):
    m = re.search(r"dataset_(\d+)$", path.name)
    return int(m.group(1)) if m else 10**9


def get_by_key(d, key):
    if key in d:
        return d[key]
    skey = str(key)
    if skey in d:
        return d[skey]
    raise KeyError(f"Cannot find key {key!r} in cluster package.")


def load_class_order(project_root: Path, spec: str, class_file: str | None):
    if class_file is not None:
        p = Path(class_file)
    else:
        if spec == "woof":
            p = project_root / "misc" / "class_woof.txt"
        elif spec == "nette":
            p = project_root / "misc" / "class_nette.txt"
        else:
            raise ValueError("For spec other than woof/nette, please pass --class-file.")

    if not p.exists():
        raise FileNotFoundError(f"Class file not found: {p}")

    classes = [x.strip() for x in p.read_text().splitlines() if x.strip()]
    return classes


class ImagePathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = [Path(p) for p in paths]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            x = self.transform(img)
        return x, str(path)


@torch.inference_mode()
def extract_features(model, transform, paths, device, batch_size=128, workers=4):
    paths = [Path(p) for p in paths]
    if len(paths) == 0:
        return torch.empty((0, 512), dtype=torch.float32)

    loader = DataLoader(
        ImagePathDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
    )

    feats = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        f = model(x)
        f = F.normalize(f.float(), p=2, dim=1)
        feats.append(f.cpu())

    return torch.cat(feats, dim=0)


def list_real_images(real_root: Path, class_name: str):
    class_dir = real_root / "train" / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Real training class directory not found: {class_dir}")
    paths = sorted(
        p for p in class_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if not paths:
        raise RuntimeError(f"No real images found in: {class_dir}")
    return paths


def resolve_slot_reference(saved_path, real_root: Path, class_name: str):
    """Resolve an original V3 reference path, even if its absolute prefix changed."""
    p = Path(saved_path)
    if p.exists():
        return p

    candidate = real_root / "train" / class_name / p.name
    if candidate.exists():
        return candidate

    # Recursive basename fallback for unusual dataset layouts.
    hits = list((real_root / "train" / class_name).rglob(p.name))
    if len(hits) == 1:
        return hits[0]

    raise FileNotFoundError(
        f"Cannot resolve V3 slot prototype path:\n"
        f"  saved: {saved_path}\n"
        f"  fallback: {candidate}"
    )


def build_resnet18(device):
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    return model, weights.transforms()


def compute_class_centroids(
    model,
    transform,
    real_root,
    class_order,
    device,
    batch_size,
    workers,
):
    centroids = {}
    counts = {}

    for cls in class_order:
        paths = list_real_images(real_root, cls)
        print(f"[real centroid] {cls}: {len(paths)} images")
        feats = extract_features(
            model, transform, paths, device,
            batch_size=batch_size, workers=workers
        )
        centroid = F.normalize(feats.mean(dim=0, keepdim=True), p=2, dim=1)[0]
        centroids[cls] = centroid
        counts[cls] = len(paths)

    return centroids, counts


def main(args):
    candidate_root = Path(args.candidate_root)
    real_root = Path(args.real_root)
    output_root = Path(args.output)
    project_root = Path(args.project_root)

    if not candidate_root.exists():
        raise FileNotFoundError(candidate_root)
    if not real_root.exists():
        raise FileNotFoundError(real_root)

    dataset_dirs = sorted(
        [p for p in candidate_root.iterdir() if p.is_dir() and re.fullmatch(r"dataset_\d+", p.name)],
        key=numeric_dataset_key,
    )
    if args.num_datasets is not None:
        dataset_dirs = dataset_dirs[: args.num_datasets]

    if len(dataset_dirs) < 2:
        raise RuntimeError(
            f"Need at least 2 candidate datasets, found {len(dataset_dirs)} in {candidate_root}"
        )

    print("Candidate datasets:")
    for p in dataset_dirs:
        print(" ", p)

    cluster_file = Path(args.cluster_file) if args.cluster_file else candidate_root / "v3_modes.pkl"
    if not cluster_file.exists():
        raise FileNotFoundError(f"Cluster file not found: {cluster_file}")

    with open(cluster_file, "rb") as f:
        package = pickle.load(f)

    if not isinstance(package, dict) or "clusters_centers_path" not in package:
        raise ValueError(
            "Cluster file must be the V3 package containing 'clusters_centers_path'."
        )

    clusters_centers_path = package["clusters_centers_path"]
    class_order_all = load_class_order(project_root, args.spec, args.class_file)

    # Use only classes actually present in dataset_0, while preserving the exact
    # class-file order used by sample_mode_guidance.py.
    present = {p.name for p in dataset_dirs[0].iterdir() if p.is_dir()}
    class_order = [c for c in class_order_all if c in present]
    if not class_order:
        raise RuntimeError(
            f"No class directories in {dataset_dirs[0]} match the class file."
        )

    print(f"Classes ({len(class_order)}): {class_order}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Feature device:", device)

    model, transform = build_resnet18(device)

    # 1) Real class centroids.
    class_centroids, real_counts = compute_class_centroids(
        model=model,
        transform=transform,
        real_root=real_root,
        class_order=class_order,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    centroid_matrix = torch.stack([class_centroids[c] for c in class_order], dim=0)

    # 2) Prepare output.
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    report_rows = []
    report_json = {
        "method": "V3 multi-candidate ResNet18 discriminative selection",
        "candidate_root": str(candidate_root),
        "real_root": str(real_root),
        "cluster_file": str(cluster_file),
        "candidate_datasets": [str(p) for p in dataset_dirs],
        "score": "slot_similarity - max_other_class_similarity",
        "safety_rule": "prefer positive own-class centroid margin; fallback if none",
        "classes": {},
        "real_class_counts": real_counts,
    }

    for class_name in class_order:
        class_idx = class_order_all.index(class_name)
        class_out = output_root / class_name
        class_out.mkdir(parents=True, exist_ok=True)

        slot_files = sorted(
            [
                p for p in (dataset_dirs[0] / class_name).iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            ],
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.name,
        )

        slot_refs = get_by_key(clusters_centers_path, class_idx)
        if len(slot_refs) < len(slot_files):
            raise RuntimeError(
                f"{class_name}: cluster package has {len(slot_refs)} slot references "
                f"but dataset has {len(slot_files)} images."
            )

        report_json["classes"][class_name] = []

        for slot_pos, base_path in enumerate(slot_files):
            candidate_paths = []
            for ds in dataset_dirs:
                p = ds / class_name / base_path.name
                if not p.exists():
                    raise FileNotFoundError(
                        f"Missing aligned candidate for {class_name}/{base_path.name}: {p}"
                    )
                candidate_paths.append(p)

            slot_ref = resolve_slot_reference(
                slot_refs[slot_pos], real_root, class_name
            )

            # One batch for all candidates + the real slot prototype.
            all_paths = candidate_paths + [slot_ref]
            all_feats = extract_features(
                model, transform, all_paths, device,
                batch_size=max(len(all_paths), 1),
                workers=min(args.workers, 2),
            )
            cand_feats = all_feats[:-1]
            slot_feat = all_feats[-1]

            slot_sims = cand_feats @ slot_feat
            class_sims = cand_feats @ centroid_matrix.T

            own_idx = class_order.index(class_name)
            own_sims = class_sims[:, own_idx]

            if len(class_order) > 1:
                mask = torch.ones(len(class_order), dtype=torch.bool)
                mask[own_idx] = False
                neg_sims, neg_local_idx = class_sims[:, mask].max(dim=1)
                neg_class_names = [c for i, c in enumerate(class_order) if i != own_idx]
                neg_names = [neg_class_names[int(i)] for i in neg_local_idx]
            else:
                neg_sims = torch.zeros_like(own_sims)
                neg_names = ["none"] * len(candidate_paths)

            margins = own_sims - neg_sims
            scores = slot_sims - neg_sims

            valid = margins > 0
            if valid.any():
                valid_scores = scores.clone()
                valid_scores[~valid] = -float("inf")
                best = int(torch.argmax(valid_scores).item())
                used_fallback = False
            else:
                best = int(torch.argmax(scores).item())
                used_fallback = True

            selected_src = candidate_paths[best]
            selected_dst = class_out / base_path.name
            shutil.copy2(selected_src, selected_dst)

            candidate_details = []
            for k, p in enumerate(candidate_paths):
                candidate_details.append({
                    "dataset": dataset_dirs[k].name,
                    "path": str(p),
                    "slot_similarity": float(slot_sims[k]),
                    "own_class_similarity": float(own_sims[k]),
                    "max_other_class_similarity": float(neg_sims[k]),
                    "nearest_other_class": neg_names[k],
                    "class_margin": float(margins[k]),
                    "score": float(scores[k]),
                    "valid_positive_margin": bool(valid[k]),
                })

            selected = candidate_details[best]
            row = {
                "class": class_name,
                "slot": base_path.stem,
                "selected_dataset": selected["dataset"],
                "selected_path": str(selected_src),
                "slot_prototype": str(slot_ref),
                "slot_similarity": selected["slot_similarity"],
                "own_class_similarity": selected["own_class_similarity"],
                "max_other_class_similarity": selected["max_other_class_similarity"],
                "nearest_other_class": selected["nearest_other_class"],
                "class_margin": selected["class_margin"],
                "score": selected["score"],
                "fallback_no_positive_margin": used_fallback,
            }
            report_rows.append(row)

            report_json["classes"][class_name].append({
                **row,
                "candidates": candidate_details,
            })

            print(
                f"[select] {class_name} slot={base_path.stem} "
                f"<- {selected['dataset']} "
                f"score={selected['score']:.4f} "
                f"slot={selected['slot_similarity']:.4f} "
                f"margin={selected['class_margin']:.4f}"
                + (" [fallback]" if used_fallback else "")
            )

    # 3) Reports.
    json_path = output_root / "selection_report.json"
    json_path.write_text(json.dumps(report_json, indent=2, ensure_ascii=False))

    csv_path = output_root / "selection_report.csv"
    fieldnames = list(report_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    total_selected = sum(
        1 for p in output_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )

    print("\nDone.")
    print("Selected dataset:", output_root)
    print("Selected images:", total_selected)
    print("JSON report:", json_path)
    print("CSV report:", csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=str, required=True)
    parser.add_argument("--real-root", type=str, required=True)
    parser.add_argument("--cluster-file", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--spec", type=str, default="woof", choices=["woof", "nette", "custom"])
    parser.add_argument("--class-file", type=str, default=None)
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parent))

    parser.add_argument("--num-datasets", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    main(parser.parse_args())