#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global-coverage candidate selection for V3-generated candidate pools.

This script PRESERVES the original local discriminative selector idea:
    local_score(x) =
        cos(f(x), f(slot_prototype))
        - max_{j != c} cos(f(x), class_centroid_j)

and ADDS class-wise global residual coverage:
    coverage_gain(x | S) =
        mean_r [ max(a_r, cos(f(r), f(x))) - a_r ]

where
    a_r = max_{s in S} cos(f(r), f(s))

Selection is greedy over ALL unselected slots of the current class:
    combined_score =
        rank_norm(local_score) + rank_norm(coverage_gain)

Hard constraints:
1) every V3 slot must select exactly one image;
2) if a slot has any candidate with positive own-class margin, only those
   positive-margin candidates are eligible;
3) if a slot has no positive-margin candidate, all of its candidates are
   allowed as a fallback.

No extra tuned weighting coefficient is introduced.

Expected candidate layout:
    candidate_root/
      dataset_0/<class>/<slot>.png
      dataset_1/<class>/<slot>.png
      ...
      v3_modes.pkl

Example:
    python select_candidates_global_coverage.py \
      --candidate-root /root/autodl-tmp/v3_candidates_woof_ipc10 \
      --real-root /root/autodl-tmp/imagewoof2 \
      --cluster-file /root/autodl-tmp/v3_candidates_woof_ipc10/v3_modes.pkl \
      --spec woof \
      --output /root/autodl-tmp/v3_global_woof_ipc10 \
      --batch-size 128 \
      --workers 4 \
      --overwrite
"""

import argparse
import csv
import json
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
    """Resolve a V3 slot-reference path even when the original prefix changed."""
    p = Path(saved_path)
    if p.exists():
        return p

    candidate = real_root / "train" / class_name / p.name
    if candidate.exists():
        return candidate

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


def compute_real_feature_bank(
    model,
    transform,
    real_root,
    class_order,
    device,
    batch_size,
    workers,
):
    """
    Extract normalized ResNet18 features for every real training image.

    Returns:
        class_centroids: dict[class_name] -> [512]
        real_features:   dict[class_name] -> [N_c, 512]
        real_paths:      dict[class_name] -> list[Path]
        counts:          dict[class_name] -> int
    """
    class_centroids = {}
    real_features = {}
    real_paths = {}
    counts = {}

    for cls in class_order:
        paths = list_real_images(real_root, cls)
        print(f"[real features] {cls}: {len(paths)} images")

        feats = extract_features(
            model,
            transform,
            paths,
            device,
            batch_size=batch_size,
            workers=workers,
        )

        centroid = F.normalize(
            feats.mean(dim=0, keepdim=True), p=2, dim=1
        )[0]

        class_centroids[cls] = centroid
        real_features[cls] = feats
        real_paths[cls] = paths
        counts[cls] = len(paths)

    return class_centroids, real_features, real_paths, counts


def rank_normalize(values: torch.Tensor):
    """
    Convert arbitrary scores to [0, 1] ranks.
    Higher raw value -> higher normalized value.

    Rank normalization avoids introducing an extra manually tuned
    weighting coefficient between local score and coverage gain.
    """
    values = values.float()
    n = values.numel()

    if n == 0:
        return values
    if n == 1:
        return torch.ones_like(values)

    order = torch.argsort(values, descending=False)
    ranks = torch.empty(n, dtype=torch.float32)
    ranks[order] = torch.arange(n, dtype=torch.float32)
    ranks /= float(n - 1)
    return ranks


def build_slot_data(
    class_name,
    class_idx,
    class_order,
    dataset_dirs,
    slot_files,
    slot_refs,
    real_root,
    centroid_matrix,
    model,
    transform,
    device,
    batch_size,
    workers,
):
    """
    Precompute static local-discriminative information for every slot/candidate.
    """
    own_idx = class_order.index(class_name)

    all_candidate_paths = []
    all_slot_ref_paths = []
    candidate_path_groups = []

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

        candidate_path_groups.append(candidate_paths)
        all_candidate_paths.extend(candidate_paths)
        all_slot_ref_paths.append(slot_ref)

    # Extract all candidate features for this class in one pass.
    candidate_feats_all = extract_features(
        model,
        transform,
        all_candidate_paths,
        device,
        batch_size=batch_size,
        workers=workers,
    )

    # Extract all slot-reference features for this class in one pass.
    slot_ref_feats_all = extract_features(
        model,
        transform,
        all_slot_ref_paths,
        device,
        batch_size=batch_size,
        workers=workers,
    )

    num_candidates = len(dataset_dirs)
    slot_data = []

    for slot_pos, base_path in enumerate(slot_files):
        start = slot_pos * num_candidates
        end = start + num_candidates

        candidate_paths = candidate_path_groups[slot_pos]
        cand_feats = candidate_feats_all[start:end]
        slot_feat = slot_ref_feats_all[slot_pos]

        slot_sims = cand_feats @ slot_feat
        class_sims = cand_feats @ centroid_matrix.T
        own_sims = class_sims[:, own_idx]

        if len(class_order) > 1:
            mask = torch.ones(len(class_order), dtype=torch.bool)
            mask[own_idx] = False

            neg_sims, neg_local_idx = class_sims[:, mask].max(dim=1)
            neg_class_names = [
                c for i, c in enumerate(class_order) if i != own_idx
            ]
            neg_names = [
                neg_class_names[int(i)] for i in neg_local_idx
            ]
        else:
            neg_sims = torch.zeros_like(own_sims)
            neg_names = ["none"] * len(candidate_paths)

        margins = own_sims - neg_sims

        # Preserve the proven local selector exactly:
        #   local_score = slot similarity - strongest other-class similarity
        local_scores = slot_sims - neg_sims
        valid = margins > 0

        slot_data.append({
            "slot_pos": slot_pos,
            "base_path": base_path,
            "slot_ref": all_slot_ref_paths[slot_pos],
            "candidate_paths": candidate_paths,
            "candidate_feats": cand_feats,
            "slot_sims": slot_sims,
            "own_sims": own_sims,
            "neg_sims": neg_sims,
            "neg_names": neg_names,
            "margins": margins,
            "local_scores": local_scores,
            "valid": valid,
        })

    return slot_data


def greedy_global_coverage_select(slot_data, real_feats):
    """
    Greedily select exactly one candidate from every slot.

    At each step:
      1) enumerate candidates from all still-unselected slots;
      2) keep the old positive-margin safety rule per slot;
      3) compute residual coverage gain over all real samples of this class;
      4) rank-normalize local score and coverage gain;
      5) select the globally best candidate;
      6) mark its slot as completed and update real-data coverage.
    """
    num_real = real_feats.shape[0]

    # Cosine similarity is in [-1, 1]. Starting from -1 means "nothing covered".
    current_coverage = torch.full(
        (num_real,), -1.0, dtype=torch.float32
    )

    unselected_slots = set(range(len(slot_data)))
    selected = {}

    for step in range(len(slot_data)):
        pool = []

        for slot_pos in sorted(unselected_slots):
            data = slot_data[slot_pos]
            valid = data["valid"]

            # Preserve original safety behavior:
            # if this slot has any positive-margin candidate, ignore invalid ones;
            # otherwise allow all candidates as fallback.
            if bool(valid.any()):
                eligible = torch.where(valid)[0].tolist()
                slot_fallback = False
            else:
                eligible = list(range(len(data["candidate_paths"])))
                slot_fallback = True

            for cand_idx in eligible:
                cand_feat = data["candidate_feats"][cand_idx]

                # Similarity from this candidate to every real sample in its class.
                real_sims = real_feats @ cand_feat

                improved = torch.maximum(current_coverage, real_sims)
                coverage_gain = (improved - current_coverage).mean()

                pool.append({
                    "slot_pos": slot_pos,
                    "cand_idx": cand_idx,
                    "local_score": float(
                        data["local_scores"][cand_idx].item()
                    ),
                    "coverage_gain": float(coverage_gain.item()),
                    "real_sims": real_sims,
                    "slot_fallback": slot_fallback,
                })

        if not pool:
            raise RuntimeError("Global selection pool unexpectedly became empty.")

        local_values = torch.tensor(
            [x["local_score"] for x in pool], dtype=torch.float32
        )
        coverage_values = torch.tensor(
            [x["coverage_gain"] for x in pool], dtype=torch.float32
        )

        local_rank = rank_normalize(local_values)
        coverage_rank = rank_normalize(coverage_values)
        combined = local_rank + coverage_rank

        best_pool_idx = int(torch.argmax(combined).item())
        best = pool[best_pool_idx]

        slot_pos = best["slot_pos"]
        cand_idx = best["cand_idx"]

        coverage_before_mean = float(current_coverage.mean().item())
        current_coverage = torch.maximum(
            current_coverage, best["real_sims"]
        )
        coverage_after_mean = float(current_coverage.mean().item())

        selected[slot_pos] = {
            "selection_step": step,
            "cand_idx": cand_idx,
            "local_score": best["local_score"],
            "coverage_gain": best["coverage_gain"],
            "local_rank_score": float(local_rank[best_pool_idx].item()),
            "coverage_rank_score": float(coverage_rank[best_pool_idx].item()),
            "combined_score": float(combined[best_pool_idx].item()),
            "fallback_no_positive_margin": best["slot_fallback"],
            "coverage_before_mean": coverage_before_mean,
            "coverage_after_mean": coverage_after_mean,
        }

        unselected_slots.remove(slot_pos)

    return selected


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
        [
            p for p in candidate_root.iterdir()
            if p.is_dir() and re.fullmatch(r"dataset_\d+", p.name)
        ],
        key=numeric_dataset_key,
    )

    if args.num_datasets is not None:
        dataset_dirs = dataset_dirs[:args.num_datasets]

    if len(dataset_dirs) < 2:
        raise RuntimeError(
            f"Need at least 2 candidate datasets, found "
            f"{len(dataset_dirs)} in {candidate_root}"
        )

    print("Candidate datasets:")
    for p in dataset_dirs:
        print(" ", p)

    cluster_file = (
        Path(args.cluster_file)
        if args.cluster_file
        else candidate_root / "v3_modes.pkl"
    )

    if not cluster_file.exists():
        raise FileNotFoundError(f"Cluster file not found: {cluster_file}")

    with open(cluster_file, "rb") as f:
        package = pickle.load(f)

    if (
        not isinstance(package, dict)
        or "clusters_centers_path" not in package
    ):
        raise ValueError(
            "Cluster file must be the V3 package containing "
            "'clusters_centers_path'."
        )

    clusters_centers_path = package["clusters_centers_path"]
    class_order_all = load_class_order(
        project_root, args.spec, args.class_file
    )

    present = {
        p.name for p in dataset_dirs[0].iterdir() if p.is_dir()
    }
    class_order = [
        c for c in class_order_all if c in present
    ]

    if not class_order:
        raise RuntimeError(
            f"No class directories in {dataset_dirs[0]} "
            f"match the class file."
        )

    print(f"Classes ({len(class_order)}): {class_order}")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )
    print("Feature device:", device)

    model, transform = build_resnet18(device)

    # ----------------------------------------------------------
    # 1) Real feature bank:
    #    - class centroids for the proven local discriminative score
    #    - all real features for the NEW residual-coverage objective
    # ----------------------------------------------------------
    (
        class_centroids,
        real_features,
        _real_paths,
        real_counts,
    ) = compute_real_feature_bank(
        model=model,
        transform=transform,
        real_root=real_root,
        class_order=class_order,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    centroid_matrix = torch.stack(
        [class_centroids[c] for c in class_order], dim=0
    )

    # ----------------------------------------------------------
    # 2) Prepare output
    # ----------------------------------------------------------
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    report_rows = []
    report_json = {
        "method": (
            "V3 multi-candidate ResNet18 local discriminative selection "
            "+ global residual coverage"
        ),
        "candidate_root": str(candidate_root),
        "real_root": str(real_root),
        "cluster_file": str(cluster_file),
        "candidate_datasets": [str(p) for p in dataset_dirs],
        "local_score": (
            "slot_similarity - max_other_class_similarity"
        ),
        "coverage_gain": (
            "mean_r[max(current_coverage_r, sim(real_r, candidate)) "
            "- current_coverage_r]"
        ),
        "combined_score": (
            "rank_norm(local_score) + rank_norm(coverage_gain)"
        ),
        "safety_rule": (
            "per slot, prefer positive own-class centroid margin; "
            "fallback to all candidates only if none is positive"
        ),
        "slot_constraint": "exactly one selected image per V3 slot",
        "classes": {},
        "real_class_counts": real_counts,
    }

    # ----------------------------------------------------------
    # 3) Class-wise global greedy selection
    # ----------------------------------------------------------
    for class_name in class_order:
        print(f"\n===== {class_name} =====")

        class_idx = class_order_all.index(class_name)
        class_out = output_root / class_name
        class_out.mkdir(parents=True, exist_ok=True)

        slot_files = sorted(
            [
                p for p in (dataset_dirs[0] / class_name).iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            ],
            key=lambda p: (
                int(p.stem) if p.stem.isdigit() else p.name
            ),
        )

        slot_refs = get_by_key(
            clusters_centers_path, class_idx
        )

        if len(slot_refs) < len(slot_files):
            raise RuntimeError(
                f"{class_name}: cluster package has {len(slot_refs)} "
                f"slot references but dataset has "
                f"{len(slot_files)} images."
            )

        slot_data = build_slot_data(
            class_name=class_name,
            class_idx=class_idx,
            class_order=class_order,
            dataset_dirs=dataset_dirs,
            slot_files=slot_files,
            slot_refs=slot_refs,
            real_root=real_root,
            centroid_matrix=centroid_matrix,
            model=model,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )

        selected = greedy_global_coverage_select(
            slot_data=slot_data,
            real_feats=real_features[class_name],
        )

        report_json["classes"][class_name] = []

        # Save in slot order so output naming/layout stays identical
        # to the original selector.
        for slot_pos, data in enumerate(slot_data):
            sel = selected[slot_pos]
            cand_idx = sel["cand_idx"]

            selected_src = data["candidate_paths"][cand_idx]
            selected_dst = class_out / data["base_path"].name
            shutil.copy2(selected_src, selected_dst)

            candidate_details = []
            for k, p in enumerate(data["candidate_paths"]):
                candidate_details.append({
                    "dataset": dataset_dirs[k].name,
                    "path": str(p),
                    "slot_similarity": float(
                        data["slot_sims"][k].item()
                    ),
                    "own_class_similarity": float(
                        data["own_sims"][k].item()
                    ),
                    "max_other_class_similarity": float(
                        data["neg_sims"][k].item()
                    ),
                    "nearest_other_class": data["neg_names"][k],
                    "class_margin": float(
                        data["margins"][k].item()
                    ),
                    "local_score": float(
                        data["local_scores"][k].item()
                    ),
                    "valid_positive_margin": bool(
                        data["valid"][k].item()
                    ),
                    "selected": (k == cand_idx),
                })

            row = {
                "class": class_name,
                "slot": data["base_path"].stem,
                "selection_step": sel["selection_step"],
                "selected_dataset": dataset_dirs[cand_idx].name,
                "selected_path": str(selected_src),
                "slot_prototype": str(data["slot_ref"]),
                "slot_similarity": float(
                    data["slot_sims"][cand_idx].item()
                ),
                "own_class_similarity": float(
                    data["own_sims"][cand_idx].item()
                ),
                "max_other_class_similarity": float(
                    data["neg_sims"][cand_idx].item()
                ),
                "nearest_other_class": data["neg_names"][cand_idx],
                "class_margin": float(
                    data["margins"][cand_idx].item()
                ),
                "local_score": sel["local_score"],
                "coverage_gain": sel["coverage_gain"],
                "local_rank_score": sel["local_rank_score"],
                "coverage_rank_score": sel["coverage_rank_score"],
                "combined_score": sel["combined_score"],
                "coverage_before_mean": sel["coverage_before_mean"],
                "coverage_after_mean": sel["coverage_after_mean"],
                "fallback_no_positive_margin": (
                    sel["fallback_no_positive_margin"]
                ),
            }

            report_rows.append(row)

            report_json["classes"][class_name].append({
                **row,
                "candidates": candidate_details,
            })

            print(
                f"[global-select step={sel['selection_step']:02d}] "
                f"{class_name} slot={data['base_path'].stem} "
                f"<- {dataset_dirs[cand_idx].name} "
                f"local={sel['local_score']:.4f} "
                f"gain={sel['coverage_gain']:.6f} "
                f"combined={sel['combined_score']:.4f}"
                + (
                    " [fallback]"
                    if sel["fallback_no_positive_margin"]
                    else ""
                )
            )

    # ----------------------------------------------------------
    # 4) Reports
    # ----------------------------------------------------------
    json_path = output_root / "selection_report.json"
    json_path.write_text(
        json.dumps(report_json, indent=2, ensure_ascii=False)
    )

    csv_path = output_root / "selection_report.csv"
    if report_rows:
        fieldnames = list(report_rows[0].keys())
        with open(
            csv_path, "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(report_rows)

    total_selected = sum(
        1
        for p in output_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )

    print("\nDone.")
    print("Selected dataset:", output_root)
    print("Selected images:", total_selected)
    print("JSON report:", json_path)
    print("CSV report:", csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidate-root", type=str, required=True
    )
    parser.add_argument(
        "--real-root", type=str, required=True
    )
    parser.add_argument(
        "--cluster-file", type=str, default=None
    )
    parser.add_argument(
        "--output", type=str, required=True
    )

    parser.add_argument(
        "--spec",
        type=str,
        default="woof",
        choices=["woof", "nette", "custom"],
    )
    parser.add_argument(
        "--class-file", type=str, default=None
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parent),
    )

    parser.add_argument(
        "--num-datasets", type=int, default=None
    )
    parser.add_argument(
        "--batch-size", type=int, default=128
    )
    parser.add_argument(
        "--workers", type=int, default=4
    )
    parser.add_argument(
        "--cpu", action="store_true"
    )
    parser.add_argument(
        "--overwrite", action="store_true"
    )

    main(parser.parse_args())

