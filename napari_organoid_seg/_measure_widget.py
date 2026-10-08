"""napari action for measuring and exporting organoid morphology."""

from pathlib import Path

import numpy as np
from napari.layers import Labels
from napari.utils.notifications import show_info

from measurements import (
    CSV_COLUMNS,
    measure_instances,
    measurement_summary,
    write_measurements_csv,
)


def _as_napari_features(
    measurements: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Build a Labels.features table with an explicit background row."""
    labels = measurements["label"]

    features: dict[str, np.ndarray] = {
        "index": np.concatenate(
            (np.array([0], dtype=np.int64), labels.astype(np.int64, copy=False))
        )
    }

    for column in CSV_COLUMNS:
        if column == "label":
            continue
        values = np.asarray(measurements[column], dtype=np.float64)
        features[column] = np.concatenate(
            (np.array([np.nan], dtype=np.float64), values)
        )

    return features


def measure_organoids(
    labels: Labels,
    output_csv: Path = Path("organoid_measurements.csv"),
) -> None:
    """Measure one Labels layer, attach features, and export a CSV."""
    data = np.asarray(labels.data)
    measurements = measure_instances(data)

    labels.features = _as_napari_features(measurements)
    output_path = write_measurements_csv(measurements, output_csv)
    summary = measurement_summary(measurements)

    count = summary["count"]
    if count == 0:
        show_info(f"No organoids found. Empty CSV written to {output_path}.")
        return

    show_info(
        f"{count} organoids | "
        f"median area {summary['median_area_px']:.1f} px² | "
        f"median diameter {summary['median_equivalent_diameter_px']:.1f} px | "
        f"median circularity {summary['median_circularity']:.3f} | "
        f"CSV: {output_path}"
    )
