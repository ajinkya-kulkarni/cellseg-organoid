"""Create a 256x256 OrganoIDNetData patch dataset.

This pools the original annotations and original test images into one image-level
source set, creates a new 70/15/15 split stratified by organoid-content
properties, then extracts 256x256 patches. Empty patches are kept so validation
and test metrics include background-only behavior.
"""

import argparse
import csv
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.io import savemat
from skimage.segmentation import relabel_sequential
from tqdm import tqdm

DEFAULT_SOURCE_ZIP = "/Users/ajinkyakulkarni/Desktop/OrganoIDNetData-All.zip"
DEFAULT_OUTPUT_DIR = "/Users/ajinkyakulkarni/Desktop/OrganoIDNetData-256"

PATCH_SIZE = 256
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

COUNT_BINS = 3
FG_BINS = 3
AREA_BINS = 2
SEED = 42


def patch_starts(dim_size):
    if dim_size <= PATCH_SIZE:
        return [0]
    starts = list(range(0, dim_size - PATCH_SIZE + 1, PATCH_SIZE))
    if starts[-1] + PATCH_SIZE < dim_size:
        starts.append(dim_size - PATCH_SIZE)
    return starts


def read_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def clean_mask(mask):
    mask = mask.copy()
    mask[mask < 0] = 0
    mask = relabel_sequential(mask)[0]
    return mask.astype(np.int32)


def image_properties(mask_path):
    mask = clean_mask(read_mask(mask_path))
    organoid_count = int(mask.max())
    instance_areas = np.bincount(mask.ravel())[1:]
    median_instance_area = float(np.median(instance_areas)) if organoid_count else 0.0
    return organoid_count, float((mask > 0).mean()), median_instance_area


def quantile_bins(values, n_bins):
    if len(values) == 0:
        return []
    if n_bins <= 1:
        return [0] * len(values)

    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    edges = np.unique(edges)
    return np.digitize(values, edges, right=True).astype(int).tolist()


def add_strata(images):
    count_values = np.log1p([img["organoid_count"] for img in images])
    fg_values = np.array([img["fg_fraction"] for img in images])
    area_values = np.log1p([img["median_instance_area"] for img in images])
    count_bins = quantile_bins(count_values, COUNT_BINS)
    fg_bins = quantile_bins(fg_values, FG_BINS)
    area_bins = quantile_bins(area_values, AREA_BINS)

    stratified = []
    for img, count_bin, fg_bin, area_bin in zip(
        images, count_bins, fg_bins, area_bins
    ):
        entry = dict(img)
        entry["count_bin"] = int(count_bin)
        entry["fg_bin"] = int(fg_bin)
        entry["area_bin"] = int(area_bin)
        entry["stratum"] = f"count{count_bin}_fg{fg_bin}_area{area_bin}"
        stratified.append(entry)
    return stratified


def rounded_count(n, ratio):
    return int(np.floor(n * ratio + 0.5))


def stratified_take(groups, target_count):
    remaining_total = sum(len(entries) for entries in groups.values())
    target_count = min(target_count, remaining_total)
    if target_count <= 0:
        return []

    ratio = target_count / remaining_total
    allocations = {}
    remainders = []
    for key in sorted(groups):
        exact = len(groups[key]) * ratio
        base = min(len(groups[key]), int(np.floor(exact)))
        allocations[key] = base
        remainders.append((exact - base, key))

    current = sum(allocations.values())
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if current >= target_count:
            break
        if allocations[key] < len(groups[key]):
            allocations[key] += 1
            current += 1

    selected = []
    for key in sorted(groups):
        take = allocations[key]
        selected.extend(groups[key][:take])
        groups[key] = groups[key][take:]
    return selected


def split_image_list(images):
    entries = add_strata(images)
    rng = np.random.default_rng(SEED)

    groups = defaultdict(list)
    for entry in entries:
        groups[entry["stratum"]].append(entry)
    for group_entries in groups.values():
        rng.shuffle(group_entries)

    n_total = len(entries)
    n_test = rounded_count(n_total, TEST_RATIO)
    n_val = rounded_count(n_total, VAL_RATIO)

    test = stratified_take(groups, n_test)
    val = stratified_take(groups, n_val)
    train = [entry for key in sorted(groups) for entry in groups[key]]

    return (
        sorted(train, key=lambda entry: entry["stem"]),
        sorted(val, key=lambda entry: entry["stem"]),
        sorted(test, key=lambda entry: entry["stem"]),
    )


def distribution(values):
    if not values:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None}
    quantiles = np.quantile(values, [0, 0.25, 0.5, 0.75, 1])
    return {
        "min": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "max": float(quantiles[4]),
    }


def split_summary(entries):
    return {
        "source_images": len(entries),
        "organoid_count": distribution([entry["organoid_count"] for entry in entries]),
        "foreground_fraction": distribution([entry["fg_fraction"] for entry in entries]),
        "median_instance_area": distribution(
            [entry["median_instance_area"] for entry in entries]
        ),
        "strata": dict(sorted(Counter(entry["stratum"] for entry in entries).items())),
    }


def write_manifest(output_dir, splits):
    manifest_path = output_dir / "split_manifest.csv"
    fieldnames = [
        "split",
        "stem",
        "source",
        "organoid_count",
        "foreground_fraction",
        "median_instance_area",
        "count_bin",
        "fg_bin",
        "area_bin",
        "stratum",
        "image_source",
        "mask_source",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, entries in splits.items():
            for entry in entries:
                writer.writerow(
                    {
                        "split": split_name,
                        "stem": entry["stem"],
                        "source": entry["source"],
                        "organoid_count": entry["organoid_count"],
                        "foreground_fraction": f"{entry['fg_fraction']:.8f}",
                        "median_instance_area": f"{entry['median_instance_area']:.3f}",
                        "count_bin": entry["count_bin"],
                        "fg_bin": entry["fg_bin"],
                        "area_bin": entry["area_bin"],
                        "stratum": entry["stratum"],
                        "image_source": entry["image_source"],
                        "mask_source": entry["mask_source"],
                    }
                )


def add_image_entry(
    images, stem, img_path, mask_path, source, image_source=None, mask_source=None
):
    organoid_count, fg_fraction, median_instance_area = image_properties(mask_path)
    images.append(
        {
            "stem": stem,
            "img_path": str(img_path),
            "mask_path": str(mask_path),
            "image_source": str(image_source or img_path),
            "mask_source": str(mask_source or mask_path),
            "source": source,
            "organoid_count": organoid_count,
            "fg_fraction": fg_fraction,
            "median_instance_area": median_instance_area,
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", default=DEFAULT_SOURCE_ZIP,
                        help="Path to zip containing Images/ and Masks/ subdirectories")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    source_zip = Path(args.source_zip)
    output_dir = Path(args.output_dir)

    if not source_zip.exists():
        raise FileNotFoundError(f"Missing source zip: {source_zip}")

    if output_dir.exists():
        print(f"Removing existing {output_dir}...")
        shutil.rmtree(output_dir)

    extract_dir = output_dir / "_extracted"
    print(f"Extracting {source_zip.name}...")
    with zipfile.ZipFile(source_zip) as z:
        z.extractall(extract_dir)

    content = list(extract_dir.iterdir())
    base_dir = content[0] if len(content) == 1 and content[0].is_dir() else extract_dir

    images = []
    missing_masks = []

    for img_path in sorted((base_dir / "Images").glob("*.tif")):
        stem = img_path.stem
        for suffix in ("_mask", ""):
            mask_path = base_dir / "Masks" / f"{stem}{suffix}.tif"
            if mask_path.exists():
                break
        if mask_path.exists():
            add_image_entry(
                images,
                stem,
                img_path,
                mask_path,
                "source_zip",
                image_source=f"{source_zip}::{img_path.name}",
                mask_source=f"{source_zip}::{mask_path.name}",
            )
        else:
            missing_masks.append((img_path, base_dir / "Masks" / f"{stem}_mask.tif"))

    if missing_masks:
        examples = "\n".join(
            f"  image={img_path} mask={mask_path}" for img_path, mask_path in missing_masks[:20]
        )
        raise RuntimeError(f"Missing masks for {len(missing_masks)} images:\n{examples}")

    print(f"Total images collected: {len(images)}")

    train_imgs, val_imgs, test_imgs = split_image_list(images)
    splits = {
        "Training": train_imgs,
        "Validation": val_imgs,
        "Test": test_imgs,
    }
    write_manifest(output_dir, splits)

    print(
        f"Split: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}"
    )

    summary = {
        "patch_size": PATCH_SIZE,
        "ratios": {"train": TRAIN_RATIO, "validation": VAL_RATIO, "test": TEST_RATIO},
        "stratification": {
            "properties": [
                "source-image organoid_count",
                "source-image foreground_fraction",
                "source-image median_instance_area",
            ],
            "count_bins": COUNT_BINS,
            "foreground_fraction_bins": FG_BINS,
            "median_instance_area_bins": AREA_BINS,
        },
        "splits": {split_name: split_summary(entries) for split_name, entries in splits.items()},
    }

    for split_name, split_imgs in splits.items():
        out_img_dir = output_dir / split_name / "images"
        out_label_dir = output_dir / split_name / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        total_patches = 0
        empty_patches = 0

        for entry in tqdm(split_imgs, desc=f"Patching {split_name}"):
            stem = entry["stem"]
            img = cv2.imread(entry["img_path"], cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Could not read image: {entry['img_path']}")
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mask = read_mask(entry["mask_path"])
            height, width = mask.shape

            x_starts = patch_starts(width)
            y_starts = patch_starts(height)

            for y in y_starts:
                for x in x_starts:
                    img_patch = img[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
                    mask_patch = mask[y : y + PATCH_SIZE, x : x + PATCH_SIZE]

                    if img_patch.shape[0] != PATCH_SIZE or img_patch.shape[1] != PATCH_SIZE:
                        continue

                    mask_patch = clean_mask(mask_patch)
                    if mask_patch.max() == 0:
                        empty_patches += 1

                    type_map = (mask_patch > 0).astype(np.int32)
                    patch_name = f"{stem}_x{x}_y{y}"

                    cv2.imwrite(
                        str(out_img_dir / f"{patch_name}.png"),
                        cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR),
                    )

                    savemat(
                        str(out_label_dir / f"{patch_name}.mat"),
                        {"inst": mask_patch, "type": type_map},
                    )

                    total_patches += 1

        split_info = summary["splits"][split_name]
        split_info["patches"] = total_patches
        split_info["empty_patches"] = empty_patches
        split_info["nonempty_patches"] = total_patches - empty_patches

        print(
            f"  {split_name}: {total_patches} patches, {empty_patches} empty included"
        )

    with (output_dir / "split_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    shutil.rmtree(extract_dir)
    print(f"\nDone. Dataset saved to {output_dir}")


if __name__ == "__main__":
    main()
