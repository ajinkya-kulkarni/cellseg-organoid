#!/usr/bin/env python3
"""Prepare an EVICAN HCT116 + HT29 whole-cell instance-segmentation dataset.

Expected EVICAN layout (original downloaded files may keep EVICAN60 in the
annotation filenames; renamed files are also supported):

    EVICAN/
      annotations/
        ...train2019....json
        ...val2019....json
        ...eval2019...easy....json
        ...eval2019...medium....json
        ...eval2019...difficult....json
      train2019/
      val2019/
      eval2019/

Only whole-cell annotations for HCT116 and HT29 are used; nucleus annotations
and all other cell lines are ignored.

EVICAN train/validation images are only partially annotated. Following the
published EVICAN preprocessing, pixels outside annotated cells plus a 10-pixel
border are Gaussian-blurred (sigma=30) so visible but unlabeled cells are not
treated as ordinary background. Source images are split before patching to
avoid leakage. Official fully annotated evaluation images are left unblurred
and used as the Test split when available.

Output matches the current cellseg-organoid training contract:

    output/
      Training/images/*.png
      Training/labels/*.mat   # keys: inst, type
      Validation/...
      Test/...
      patch_manifest.csv
      split_summary.json
"""

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.io import savemat
from skimage.segmentation import relabel_sequential
from tqdm import tqdm


PATCH_SIZE = 256
VAL_RATIO = 0.15
SEED = 42
BLUR_SIGMA = 30.0
ANNOTATION_BORDER = 10
TARGET_CELL_LINES = ("HCT116", "HT29")

_NUCLEUS_WORDS = ("nuc", "nucleus", "nuclei", "nuclear")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evican-root",
        type=Path,
        required=True,
        help="EVICAN directory containing annotations/, train2019/, val2019/, eval2019/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to create with Training/Validation/Test patches.",
    )
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help="Fraction of pooled partial train+val source images held out for Validation.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--blur-sigma", type=float, default=BLUR_SIGMA)
    parser.add_argument("--annotation-border", type=int, default=ANNOTATION_BORDER)
    parser.add_argument(
        "--keep-empty-partial",
        action="store_true",
        help="Also save empty patches from partially annotated train/validation images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output-dir if it already exists.",
    )
    return parser.parse_args()


def normalize_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def category_map(categories):
    """Map EVICAN60 whole-cell category IDs to HCT116/HT29."""
    out = {}
    for category in categories:
        name = str(category.get("name", ""))
        norm = normalize_name(name)
        if any(word in norm for word in _NUCLEUS_WORDS):
            continue
        for cell_line in TARGET_CELL_LINES:
            if normalize_name(cell_line) in norm:
                out[int(category["id"])] = cell_line
                break
    return out


def read_json(path):
    with path.open() as f:
        return json.load(f)


def discover_annotation_files(evican_root, subset):
    annotations_dir = evican_root / "annotations"
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Missing annotations directory: {annotations_dir}")

    subset_pattern = re.compile(
        rf"(^|[^a-z0-9]){re.escape(subset.lower())}2019([^a-z0-9]|$)"
    )
    candidates = [
        path
        for path in sorted(annotations_dir.glob("*.json"))
        if subset_pattern.search(path.name.lower())
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No {subset}2019 annotation JSONs found in {annotations_dir}"
        )

    matched = []
    for path in candidates:
        data = read_json(path)
        mapping = category_map(data.get("categories", []))
        if mapping:
            matched.append((path, data, mapping))

    if not matched:
        names = ", ".join(path.name for path in candidates)
        raise RuntimeError(
            f"Found {subset} JSONs but none look like EVICAN60 HCT116/HT29 annotations: {names}"
        )

    # Prefer explicitly named EVICAN60 files when both EVICAN2 and EVICAN60 are present.
    explicit = [item for item in matched if "evican60" in item[0].name.lower()]
    if explicit:
        matched = explicit

    if subset in {"train", "val"}:
        both_lines = [
            item for item in matched if set(item[2].values()) == set(TARGET_CELL_LINES)
        ]
        if both_lines:
            matched = both_lines
        if len(matched) != 1:
            names = ", ".join(item[0].name for item in matched)
            raise RuntimeError(
                f"Expected one EVICAN60 {subset} annotation JSON, found: {names}"
            )

    return matched


def index_image_dir(image_dir):
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    index = {}
    for path in image_dir.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, path)
    return index


def selected_entries(annotation_file, data, cat_to_line, image_dir, source_subset):
    images = {int(item["id"]): item for item in data.get("images", [])}
    annotations_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        category_id = int(ann["category_id"])
        if category_id in cat_to_line:
            annotations_by_image[int(ann["image_id"])].append(ann)

    image_index = index_image_dir(image_dir)
    entries = []
    for image_id, annotations in annotations_by_image.items():
        image_info = images.get(image_id)
        if image_info is None:
            raise RuntimeError(
                f"{annotation_file.name}: annotation references missing image id {image_id}"
            )

        lines = {cat_to_line[int(ann["category_id"])] for ann in annotations}
        if len(lines) != 1:
            raise RuntimeError(
                f"{annotation_file.name}: image id {image_id} contains selected annotations "
                f"from multiple cell lines: {sorted(lines)}"
            )
        cell_line = next(iter(lines))

        file_name = Path(str(image_info["file_name"])).name
        image_path = image_dir / str(image_info["file_name"])
        if not image_path.exists():
            image_path = image_index.get(file_name)
        if image_path is None or not image_path.exists():
            raise FileNotFoundError(
                f"{annotation_file.name}: could not find image {image_info['file_name']} "
                f"under {image_dir}"
            )

        entries.append(
            {
                "key": f"{source_subset}:{annotation_file.stem}:{image_id}",
                "source_subset": source_subset,
                "annotation_file": str(annotation_file),
                "image_id": image_id,
                "file_name": file_name,
                "image_path": str(image_path),
                "width": int(image_info["width"]),
                "height": int(image_info["height"]),
                "cell_line": cell_line,
                "annotations": annotations,
            }
        )
    return entries


def polygon_to_mask(segmentation, height, width):
    if not isinstance(segmentation, list):
        raise TypeError(
            "EVICAN annotations are expected to use COCO polygon segmentations; "
            f"got {type(segmentation).__name__}"
        )
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        coords = np.asarray(polygon, dtype=np.float32)
        if coords.size < 6 or coords.size % 2:
            continue
        points = np.rint(coords.reshape(-1, 2)).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def rasterize_instances(entry, image_shape):
    height, width = image_shape[:2]
    if height != entry["height"] or width != entry["width"]:
        raise RuntimeError(
            f"Shape mismatch for {entry['file_name']}: image={width}x{height}, "
            f"annotation={entry['width']}x{entry['height']}"
        )

    instance_map = np.zeros((height, width), dtype=np.int32)
    overlap_pixels = 0
    next_id = 1
    for ann in sorted(entry["annotations"], key=lambda item: int(item["id"])):
        binary = polygon_to_mask(ann.get("segmentation", []), height, width)
        overlap_pixels += int(np.count_nonzero(binary & (instance_map > 0)))
        binary &= instance_map == 0
        if not binary.any():
            continue
        instance_map[binary] = next_id
        next_id += 1

    instance_map = relabel_sequential(instance_map)[0].astype(np.int32)
    return instance_map, overlap_pixels


def blur_unannotated(image, instance_map, sigma, border):
    if sigma <= 0:
        return image.copy()
    annotated = (instance_map > 0).astype(np.uint8)
    if border > 0:
        kernel_size = 2 * border + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        annotated = cv2.dilate(annotated, kernel, iterations=1)

    blurred = cv2.GaussianBlur(
        image,
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    keep = annotated.astype(bool)
    out = blurred
    out[keep] = image[keep]
    return out


def patch_starts(dim_size, patch_size):
    if dim_size <= patch_size:
        return [0]
    starts = list(range(0, dim_size - patch_size + 1, patch_size))
    if starts[-1] + patch_size < dim_size:
        starts.append(dim_size - patch_size)
    return starts


def split_partial_entries(entries, val_ratio, seed):
    if not 0 <= val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1).")

    rng = np.random.default_rng(seed)
    train = []
    val = []
    for cell_line in TARGET_CELL_LINES:
        group = [entry for entry in entries if entry["cell_line"] == cell_line]
        group = sorted(group, key=lambda item: item["key"])
        rng.shuffle(group)
        if len(group) <= 1 or val_ratio == 0:
            n_val = 0
        else:
            n_val = max(1, int(round(len(group) * val_ratio)))
            n_val = min(n_val, len(group) - 1)
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    return (
        sorted(train, key=lambda item: item["key"]),
        sorted(val, key=lambda item: item["key"]),
    )


def cell_line_counts(entries):
    return {
        cell_line: {
            "source_images": sum(entry["cell_line"] == cell_line for entry in entries),
            "annotated_instances": sum(
                len(entry["annotations"])
                for entry in entries
                if entry["cell_line"] == cell_line
            ),
        }
        for cell_line in TARGET_CELL_LINES
    }


def safe_stem(entry):
    raw = Path(entry["file_name"]).stem
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    return f"{entry['cell_line']}_{entry['source_subset']}_{entry['image_id']}_{raw}"


def save_split(
    entries,
    split_name,
    output_dir,
    patch_size,
    partial,
    blur_sigma,
    annotation_border,
    keep_empty_partial,
    manifest_rows,
):
    out_img_dir = output_dir / split_name / "images"
    out_label_dir = output_dir / split_name / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "source_images": len(entries),
        "annotated_instances": sum(len(entry["annotations"]) for entry in entries),
        "patches": 0,
        "nonempty_patches": 0,
        "empty_patches": 0,
        "skipped_empty_partial_patches": 0,
        "polygon_overlap_pixels": 0,
        "cell_lines": cell_line_counts(entries),
    }

    for entry in tqdm(entries, desc=f"Preparing {split_name}"):
        image = cv2.imread(entry["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {entry['image_path']}")

        instance_map, overlap_pixels = rasterize_instances(entry, image.shape)
        stats["polygon_overlap_pixels"] += overlap_pixels

        prepared = (
            blur_unannotated(image, instance_map, blur_sigma, annotation_border)
            if partial
            else image
        )

        height, width = instance_map.shape
        for y in patch_starts(height, patch_size):
            for x in patch_starts(width, patch_size):
                image_patch = prepared[y : y + patch_size, x : x + patch_size]
                inst_patch = instance_map[y : y + patch_size, x : x + patch_size]

                # Pad source images smaller than the requested patch size.
                if image_patch.shape[:2] != (patch_size, patch_size):
                    pad_bottom = patch_size - image_patch.shape[0]
                    pad_right = patch_size - image_patch.shape[1]
                    image_patch = cv2.copyMakeBorder(
                        image_patch,
                        0,
                        pad_bottom,
                        0,
                        pad_right,
                        cv2.BORDER_CONSTANT,
                        value=0,
                    )
                    inst_patch = np.pad(
                        inst_patch,
                        ((0, pad_bottom), (0, pad_right)),
                        mode="constant",
                    )

                inst_patch = relabel_sequential(inst_patch)[0].astype(np.int32)
                is_empty = inst_patch.max() == 0
                if partial and is_empty and not keep_empty_partial:
                    stats["skipped_empty_partial_patches"] += 1
                    continue

                type_map = (inst_patch > 0).astype(np.int32)
                patch_name = f"{safe_stem(entry)}_x{x}_y{y}"
                cv2.imwrite(str(out_img_dir / f"{patch_name}.png"), image_patch)
                savemat(
                    str(out_label_dir / f"{patch_name}.mat"),
                    {"inst": inst_patch, "type": type_map},
                )

                n_instances = int(inst_patch.max())
                fg_fraction = float((inst_patch > 0).mean())
                manifest_rows.append(
                    {
                        "split": split_name,
                        "cell_line": entry["cell_line"],
                        "source_subset": entry["source_subset"],
                        "source_image_id": entry["image_id"],
                        "source_file": entry["file_name"],
                        "annotation_file": Path(entry["annotation_file"]).name,
                        "patch": patch_name,
                        "x": x,
                        "y": y,
                        "instances": n_instances,
                        "foreground_fraction": f"{fg_fraction:.8f}",
                        "partial_annotation": int(partial),
                        "blur_sigma": blur_sigma if partial else 0,
                        "annotation_border": annotation_border if partial else 0,
                    }
                )
                stats["patches"] += 1
                if is_empty:
                    stats["empty_patches"] += 1
                else:
                    stats["nonempty_patches"] += 1

    return stats


def main():
    args = parse_args()
    evican_root = args.evican_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if args.patch_size <= 0:
        raise ValueError("--patch-size must be > 0")
    if args.blur_sigma < 0:
        raise ValueError("--blur-sigma must be >= 0")
    if args.annotation_border < 0:
        raise ValueError("--annotation-border must be >= 0")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    train_files = discover_annotation_files(evican_root, "train")
    val_files = discover_annotation_files(evican_root, "val")
    eval_files = discover_annotation_files(evican_root, "eval")

    partial_entries = []
    annotation_inputs = []
    for subset, files in (("train", train_files), ("val", val_files)):
        path, data, cat_to_line = files[0]
        annotation_inputs.append(str(path))
        partial_entries.extend(
            selected_entries(
                path,
                data,
                cat_to_line,
                evican_root / f"{subset}2019",
                f"{subset}2019",
            )
        )

    train_entries, val_entries = split_partial_entries(
        partial_entries, args.val_ratio, args.seed
    )

    test_entries = []
    seen_test_keys = set()
    for path, data, cat_to_line in eval_files:
        annotation_inputs.append(str(path))
        for entry in selected_entries(
            path,
            data,
            cat_to_line,
            evican_root / "eval2019",
            "eval2019",
        ):
            # Difficulty JSONs should partition the evaluation set. De-duplicate
            # defensively in case the local EVICAN folder contains copies.
            dedupe_key = (entry["image_id"], entry["file_name"], entry["cell_line"])
            if dedupe_key in seen_test_keys:
                continue
            seen_test_keys.add(dedupe_key)
            test_entries.append(entry)
    test_entries = sorted(test_entries, key=lambda item: item["key"])

    if not partial_entries:
        raise RuntimeError("No HCT116/HT29 whole-cell annotations found.")

    print("Selected partial source data:")
    print(json.dumps(cell_line_counts(partial_entries), indent=2))
    print(
        f"Source split: Training={len(train_entries)} images, "
        f"Validation={len(val_entries)} images"
    )
    print("Official fully annotated evaluation data:")
    print(json.dumps(cell_line_counts(test_entries), indent=2))
    missing_test = [
        line
        for line in TARGET_CELL_LINES
        if not any(entry["cell_line"] == line for entry in test_entries)
    ]
    if missing_test:
        print(
            "WARNING: no official EVICAN evaluation images were found for: "
            + ", ".join(missing_test)
        )

    manifest_rows = []
    split_stats = {}
    split_stats["Training"] = save_split(
        train_entries,
        "Training",
        output_dir,
        args.patch_size,
        partial=True,
        blur_sigma=args.blur_sigma,
        annotation_border=args.annotation_border,
        keep_empty_partial=args.keep_empty_partial,
        manifest_rows=manifest_rows,
    )
    split_stats["Validation"] = save_split(
        val_entries,
        "Validation",
        output_dir,
        args.patch_size,
        partial=True,
        blur_sigma=args.blur_sigma,
        annotation_border=args.annotation_border,
        keep_empty_partial=args.keep_empty_partial,
        manifest_rows=manifest_rows,
    )
    split_stats["Test"] = save_split(
        test_entries,
        "Test",
        output_dir,
        args.patch_size,
        partial=False,
        blur_sigma=0,
        annotation_border=0,
        keep_empty_partial=True,
        manifest_rows=manifest_rows,
    )

    manifest_path = output_dir / "patch_manifest.csv"
    fieldnames = [
        "split",
        "cell_line",
        "source_subset",
        "source_image_id",
        "source_file",
        "annotation_file",
        "patch",
        "x",
        "y",
        "instances",
        "foreground_fraction",
        "partial_annotation",
        "blur_sigma",
        "annotation_border",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "dataset": "EVICAN60",
        "target": "whole-cell instance segmentation",
        "cell_lines": list(TARGET_CELL_LINES),
        "patch_size": args.patch_size,
        "seed": args.seed,
        "partial_source_pool": ["train2019", "val2019"],
        "partial_split": {
            "validation_ratio": args.val_ratio,
            "stratified_by": "cell_line",
            "split_before_patching": True,
        },
        "partial_annotation_preprocessing": {
            "gaussian_blur_sigma": args.blur_sigma,
            "preserved_border_pixels": args.annotation_border,
            "keep_empty_partial_patches": args.keep_empty_partial,
        },
        "test_source": "official fully annotated eval2019 images",
        "annotation_files": sorted(set(annotation_inputs)),
        "splits": split_stats,
    }
    with (output_dir / "split_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Dataset saved to {output_dir}")
    for split_name, stats in split_stats.items():
        print(
            f"{split_name}: {stats['source_images']} source images -> "
            f"{stats['patches']} patches "
            f"({stats['nonempty_patches']} nonempty, {stats['empty_patches']} empty)"
        )


if __name__ == "__main__":
    main()
