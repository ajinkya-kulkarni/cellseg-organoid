"""Per-organoid morphology measurements for 2D instance-label images."""

import csv
import math
from pathlib import Path

import numpy as np
from skimage.measure import regionprops_table

CSV_COLUMNS = (
    "label",
    "area_px",
    "equivalent_diameter_px",
    "perimeter_crofton_px",
    "circularity",
    "solidity",
    "eccentricity",
    "centroid_x_px",
    "centroid_y_px",
)


def measure_instances(label_image: np.ndarray) -> dict[str, np.ndarray]:
    """Measure the core 2D morphology of each non-zero instance label."""
    label_image = np.asarray(label_image)

    if label_image.ndim != 2:
        raise ValueError(
            f"Expected one 2D label image, got shape {label_image.shape}."
        )
    if np.any(label_image < 0):
        raise ValueError("Label images must contain non-negative integer IDs.")

    props = regionprops_table(
        label_image,
        properties=(
            "label",
            "area",
            "equivalent_diameter_area",
            "perimeter_crofton",
            "solidity",
            "eccentricity",
            "centroid",
        ),
    )

    area = np.asarray(props["area"], dtype=np.float64)
    perimeter = np.asarray(props["perimeter_crofton"], dtype=np.float64)

    circularity = np.full(area.shape, np.nan, dtype=np.float64)
    np.divide(
        4.0 * math.pi * area,
        perimeter * perimeter,
        out=circularity,
        where=perimeter > 0,
    )

    return {
        "label": np.asarray(props["label"], dtype=np.int64),
        "area_px": area,
        "equivalent_diameter_px": np.asarray(
            props["equivalent_diameter_area"], dtype=np.float64
        ),
        "perimeter_crofton_px": perimeter,
        "circularity": circularity,
        "solidity": np.asarray(props["solidity"], dtype=np.float64),
        "eccentricity": np.asarray(props["eccentricity"], dtype=np.float64),
        "centroid_x_px": np.asarray(props["centroid-1"], dtype=np.float64),
        "centroid_y_px": np.asarray(props["centroid-0"], dtype=np.float64),
    }


def write_measurements_csv(
    measurements: dict[str, np.ndarray],
    output_path: str | Path,
) -> Path:
    """Write one CSV row per organoid and return the resolved output path."""
    output_path = Path(output_path).expanduser()
    if not output_path.suffix:
        output_path = output_path.with_suffix(".csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(measurements["label"])
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row_index in range(n_rows):
            writer.writerow(
                {
                    column: measurements[column][row_index]
                    for column in CSV_COLUMNS
                }
            )

    return output_path.resolve()


def measurement_summary(
    measurements: dict[str, np.ndarray],
) -> dict[str, float | int]:
    """Return compact culture-level summary statistics."""
    count = int(len(measurements["label"]))
    if count == 0:
        return {
            "count": 0,
            "median_area_px": math.nan,
            "median_equivalent_diameter_px": math.nan,
            "median_circularity": math.nan,
        }

    return {
        "count": count,
        "median_area_px": float(np.median(measurements["area_px"])),
        "median_equivalent_diameter_px": float(
            np.median(measurements["equivalent_diameter_px"])
        ),
        "median_circularity": float(np.nanmedian(measurements["circularity"])),
    }
