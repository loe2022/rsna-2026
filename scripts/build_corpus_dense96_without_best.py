#!/usr/bin/env python3
"""Build the Max-Span Dense-96 cache used by cur_best_eval94_dense96.

The five slot proportions remain the public Raptor 18/14/12/8/12 layout, scaled
to a target of 96 real slices.  Per-study quotas are capped by the number of
unique source slices in each selected series; unused quota is redistributed to
the other slots.  Slices are sampled once from 2--98 percent of each series,
never repeated, compacted at the front of the fixed 96-plane cache, and followed
only by masked padding when a whole study physically has fewer than 96 usable
planes.

Series selection, 140 mm crop, 336-pixel cache, intensity windowing and output
file layout otherwise match the original build_corpus_80.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


IMG = 336
CROP_MM = 140.0
SLOTS = [
    ("Sagittal", 1, 27),
    ("Sagittal", 0, 21),
    ("Coronal", 1, 18),
    ("Coronal", 0, 12),
    ("Axial", -1, 18),
]
SPAN_LO, SPAN_HI = 0.04, 0.96
DEPTH = 96
DEFAULT_SPLIT = 3200

_SERIES_BY_STUDY = None
_IMAGES_DIR = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a capacity-aware Max-Span Dense-96 x 336 x 336 uint8 cache "
            "using the first matching series in train_series.csv order."
        )
    )
    parser.add_argument(
        "--competition_dir",
        default="/root/RSNA/rsna-knee-abnormality-detection",
        help="Directory containing train.csv, train_series.csv and train_images/.",
    )
    parser.add_argument(
        "--output_dir",
        default="/root/RSNA/train_data/knee-raptor-corpus-96-dense-v2",
        help="Output directory for the first corpus part.",
    )
    parser.add_argument(
        "--output_ext_dir",
        default="/root/RSNA/train_data/knee-raptor-corpus-96-dense-v2-ext",
        help="Output directory for the second corpus part.",
    )
    parser.add_argument(
        "--split", type=int, default=DEFAULT_SPLIT,
        help="Number of studies in all_*.npy; the remainder goes to extra_*.npy.",
    )
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1),
        help="Number of DICOM preprocessing worker processes.",
    )
    parser.add_argument(
        "--flush_every", type=int, default=16,
        help="Flush memmaps and save progress after this many completed studies.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Build only the first N studies for a smoke test; 0 builds all studies.",
    )
    parser.add_argument(
        "--reference_corpus_dir", default=None,
        help="Optional old part-1 corpus directory; use its IDs to preserve row order.",
    )
    parser.add_argument(
        "--reference_corpus_ext_dir", default=None,
        help="Optional old part-2 corpus directory; use its IDs to preserve row order.",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume compatible .partial.npy files and build_progress.npy (default).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete this script's existing partial/final outputs and rebuild.",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.flush_every < 1 or args.split < 1 or args.limit < 0:
        parser.error("workers/flush_every/split must be positive and limit nonnegative.")
    if bool(args.reference_corpus_dir) != bool(args.reference_corpus_ext_dir):
        parser.error("Provide both reference corpus directories or neither.")
    return args


def _check_runtime_dependencies():
    """Fail before allocating files or starting thousands of worker tasks."""
    missing = []
    try:
        import pydicom  # noqa: F401
    except ModuleNotFoundError:
        missing.append("pydicom>=2.4,<4")
    try:
        import cv2  # noqa: F401
    except ModuleNotFoundError:
        missing.append("opencv-python-headless==4.11.0.86")
    if missing:
        packages = " ".join(f'"{package}"' for package in missing)
        raise ModuleNotFoundError(
            "Missing DICOM preprocessing dependencies. Install them into the same "
            f"Python environment with:\n  {sys.executable} -m pip install {packages}"
        )


def _atomic_json(path: Path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _atomic_npy(path: Path, array):
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, array, allow_pickle=False)
    os.replace(tmp, path)


def _normalize_plane(value):
    text = str(value).strip().lower()
    return {"sagittal": "Sagittal", "coronal": "Coronal", "axial": "Axial"}.get(text, str(value).strip())


def _normalize_fluid(value):
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y"}:
        return 1
    if text in {"0", "0.0", "false", "no", "n"}:
        return 0
    return None


def _resolve_inputs(competition_dir: Path):
    train_csv = competition_dir / "train.csv"
    series_csv = competition_dir / "train_series.csv"
    images_dir = next(
        (competition_dir / name for name in ("train_images", "train_series")
         if (competition_dir / name).is_dir()),
        None,
    )
    for path in (train_csv, series_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    if images_dir is None:
        raise FileNotFoundError(
            f"No train_images/ or train_series/ directory under {competition_dir}"
        )
    return train_csv, series_csv, images_dir


def _load_ids(train_csv: Path, args):
    train = pd.read_csv(train_csv, dtype={"StudyInstanceUID": str})
    if "StudyInstanceUID" not in train.columns:
        raise ValueError(f"StudyInstanceUID is missing from {train_csv}")
    official = train["StudyInstanceUID"].astype(str).tolist()
    if len(set(official)) != len(official) or any(not uid.strip() for uid in official):
        raise ValueError("train.csv study IDs must be unique and nonempty.")

    if args.reference_corpus_dir:
        first = np.load(Path(args.reference_corpus_dir) / "all_ids.npy", allow_pickle=False).astype(str)
        second = np.load(Path(args.reference_corpus_ext_dir) / "extra_ids.npy", allow_pickle=False).astype(str)
        ids = np.concatenate([first, second]).tolist()
        if len(ids) != len(official) or set(ids) != set(official):
            raise ValueError("Reference corpus IDs do not exactly match official train.csv IDs.")
        id_source = "reference_corpus"
    else:
        ids = official
        id_source = "official_train_csv"

    if args.limit:
        ids = ids[:args.limit]
    return ids, id_source


def _load_series(series_csv: Path):
    frame = pd.read_csv(
        series_csv,
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    required = ["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing train_series.csv columns: {missing}")
    if frame["SeriesInstanceUID"].duplicated().any():
        raise ValueError("train_series.csv contains duplicate SeriesInstanceUID rows.")
    frame["Anatomical_Plane"] = frame["Anatomical_Plane"].map(_normalize_plane)
    frame["Fluid_Sensitive"] = frame["Fluid_Sensitive"].map(_normalize_fluid)
    return {
        str(uid): rows.to_dict("records")
        for uid, rows in frame.groupby("StudyInstanceUID", sort=False)
    }


def _init_worker(series_by_study, images_dir):
    global _SERIES_BY_STUDY, _IMAGES_DIR
    _SERIES_BY_STUDY = series_by_study
    _IMAGES_DIR = images_dir


def _dicom_files(series_dir: str):
    files = glob.glob(os.path.join(series_dir, "*.dcm"))
    if not files:
        files = [str(path) for path in Path(series_dir).iterdir() if path.is_file()] if os.path.isdir(series_dir) else []
    return files


def _order_and_meta(series_dir: str):
    import pydicom

    records = []
    spacings = []
    for filename in _dicom_files(series_dir):
        try:
            header = pydicom.dcmread(filename, stop_before_pixels=True, force=True)
            iop = getattr(header, "ImageOrientationPatient", None)
            ipp = getattr(header, "ImagePositionPatient", None)
            instance = float(getattr(header, "InstanceNumber", 0) or 0)
            if iop is not None and ipp is not None and len(iop) == 6 and len(ipp) >= 3:
                row = np.asarray(iop[:3], dtype=np.float64)
                col = np.asarray(iop[3:], dtype=np.float64)
                normal = np.cross(row, col)
                position = float(np.dot(np.asarray(ipp[:3], dtype=np.float64), normal))
            else:
                position = instance
            spacing = getattr(header, "PixelSpacing", None)
            spacing = float(spacing[0]) if spacing is not None and len(spacing) else 0.5
            if not np.isfinite(spacing) or spacing <= 0:
                spacing = 0.5
            spacings.append(spacing)
            records.append((position, instance, filename, spacing))
        except Exception:
            records.append((0.0, 0.0, filename, 0.5))
    records.sort(key=lambda item: (item[0], item[1], item[2]))
    median_spacing = float(np.median(spacings)) if spacings else 0.5
    return [(filename, spacing) for _, _, filename, spacing in records], median_spacing


def _read_pixels(filename: str):
    import pydicom
    try:
        from pydicom.pixels import apply_modality_lut
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_modality_lut

    dicom = pydicom.dcmread(filename, force=True)
    array = apply_modality_lut(dicom.pixel_array, dicom).astype(np.float32)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D DICOM image, got {array.shape}: {filename}")
    if str(getattr(dicom, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        array = array.max() - array
    return array


def _crop_resize(array, pixel_spacing):
    import cv2

    height, width = array.shape
    crop_pixels = int(round(CROP_MM / max(float(pixel_spacing), 1e-3)))
    crop_pixels = max(1, min(crop_pixels, height, width))
    y0 = (height - crop_pixels) // 2
    x0 = (width - crop_pixels) // 2
    cropped = array[y0:y0 + crop_pixels, x0:x0 + crop_pixels]
    # Keep this identical to baseline_v0/submit_80.py.
    return cv2.resize(cropped, (IMG, IMG), interpolation=cv2.INTER_AREA)


def _pick_series(rows, plane, fluid, used):
    candidates = [
        row for row in rows
        if row["Anatomical_Plane"] == plane and row["SeriesInstanceUID"] not in used
    ]
    if fluid in (0, 1):
        preferred = [row for row in candidates if row.get("Fluid_Sensitive") == fluid]
        if preferred:
            return preferred[0]
    return candidates[0] if candidates else None


def _eligible_bounds(length):
    """Inclusive 2--98 percent bounds, matching submission preprocessing."""
    if length < 1:
        return 0, -1
    lo = min(int(length * SPAN_LO), length - 1)
    hi = min(max(lo, int(length * SPAN_HI) - 1), length - 1)
    return lo, hi


def _dense_allocate(capacities, proportions, target):
    """Largest-remainder allocation with capacity-aware redistribution."""
    caps = np.asarray(capacities, dtype=np.int64)
    weights = np.asarray(proportions, dtype=np.float64)
    if caps.ndim != 1 or weights.shape != caps.shape:
        raise ValueError("capacities/proportions must be one-dimensional and aligned.")
    if np.any(caps < 0) or np.any(weights < 0) or target < 0:
        raise ValueError("capacities, proportions and target must be nonnegative.")
    quota = np.zeros_like(caps)
    remaining = min(int(target), int(caps.sum()))
    while remaining:
        active = np.flatnonzero(quota < caps)
        if not len(active):
            break
        active_weights = weights[active]
        if not np.any(active_weights > 0):
            active_weights = np.ones(len(active), dtype=np.float64)
        raw = remaining * active_weights / active_weights.sum()
        base = np.floor(raw).astype(np.int64)
        base = np.minimum(base, caps[active] - quota[active])
        granted = int(base.sum())
        if granted:
            quota[active] += base
            remaining -= granted
            if not remaining:
                break

        active = np.flatnonzero(quota < caps)
        if not len(active):
            break
        active_weights = weights[active]
        if not np.any(active_weights > 0):
            active_weights = np.ones(len(active), dtype=np.float64)
        raw = remaining * active_weights / active_weights.sum()
        fractions = raw - np.floor(raw)
        order = sorted(
            range(len(active)),
            key=lambda j: (-fractions[j], -active_weights[j], int(active[j])),
        )
        for j in order:
            slot = int(active[j])
            if remaining < 1:
                break
            if quota[slot] < caps[slot]:
                quota[slot] += 1
                remaining -= 1
    if int(quota.sum()) != min(int(target), int(caps.sum())):
        raise AssertionError("Dense quota allocation did not reach the feasible target.")
    return quota


def _dense_unique_indices(length, count):
    """Evenly cover the eligible span without repeating a source DICOM."""
    lo, hi = _eligible_bounds(length)
    capacity = max(0, hi - lo + 1)
    if count < 0 or count > capacity:
        raise ValueError(f"Cannot select {count} unique slices from capacity {capacity}.")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    picks = np.linspace(lo, hi, count).round().astype(np.int64)
    if len(np.unique(picks)) != count:
        raise AssertionError("Dense linspace unexpectedly produced repeated indices.")
    return picks


def _build_study(task):
    index, study_uid = task
    try:
        rows = _SERIES_BY_STUDY.get(study_uid, [])
        volume = np.zeros((DEPTH, IMG, IMG), dtype=np.uint8)
        mask = np.zeros(DEPTH, dtype=np.uint8)
        used = set()
        slot_stats = []

        selected_slots = []
        capacities = []
        proportions = []
        for plane, fluid, nominal in SLOTS:
            row = _pick_series(rows, plane, fluid, used)
            series_uid = None
            files, median_spacing = [], 0.5
            if row is not None:
                series_uid = str(row["SeriesInstanceUID"])
                used.add(series_uid)
                series_dir = os.path.join(_IMAGES_DIR, study_uid, series_uid)
                files, median_spacing = _order_and_meta(series_dir)
            lo, hi = _eligible_bounds(len(files))
            capacity = max(0, hi - lo + 1)
            selected_slots.append(
                (plane, fluid, nominal, series_uid, files, median_spacing)
            )
            capacities.append(capacity)
            proportions.append(nominal)

        quotas = _dense_allocate(capacities, proportions, DEPTH)
        packed = []
        for slot_index, slot in enumerate(selected_slots):
            plane, fluid, nominal, series_uid, files, median_spacing = slot
            quota = int(quotas[slot_index])
            stat = {
                "plane": plane,
                "fluid": fluid,
                "nominal": nominal,
                "series": series_uid,
                "dicoms": len(files),
                "eligible": int(capacities[slot_index]),
                "allocated": quota,
                "decoded": 0,
                "stored": 0,
            }
            if quota == 0:
                slot_stats.append(stat)
                continue

            picks = _dense_unique_indices(len(files), quota)
            arrays = []
            spacings = []
            for pick in picks:
                filename, spacing = files[int(pick)]
                try:
                    arrays.append(_read_pixels(filename))
                    spacings.append(spacing)
                    stat["decoded"] += 1
                except Exception:
                    arrays.append(None)
                    spacings.append(median_spacing)

            valid = [array for array in arrays if array is not None]
            if valid:
                pixels = np.concatenate([array.reshape(-1) for array in valid])
                low, high = np.percentile(pixels, [2.0, 98.0])
                del pixels
            else:
                low, high = 0.0, 1.0

            for array, spacing in zip(arrays, spacings):
                if array is None:
                    continue
                normalized = np.clip(
                    (array - low) / (high - low + 1e-6), 0.0, 1.0
                )
                resized = _crop_resize(
                    normalized, spacing if spacing > 0 else median_spacing
                )
                stored = (resized * 255.0).astype(np.uint8)
                # Keep valid planes contiguous so no 2.5D window contains padding.
                if np.any(stored != 0):
                    packed.append(stored)
                    stat["stored"] += 1
            slot_stats.append(stat)

        if len(packed) < 3:
            raise RuntimeError(
                f"Only {len(packed)} usable DICOM slices; at least 3 are required."
            )
        stored_count = min(len(packed), DEPTH)
        volume[:stored_count] = np.stack(packed[:stored_count], axis=0)
        mask[:stored_count] = 1
        return index, study_uid, volume, mask, slot_stats, None
    except Exception as exc:
        return index, study_uid, None, None, None, (
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        )


def _part_specs(total, split, output_dir: Path, output_ext_dir: Path):
    first_count = min(split, total)
    specs = [(output_dir, "all", 0, first_count)]
    if total > first_count:
        specs.append((output_ext_dir, "extra", first_count, total))
    return specs


def _generated_paths(specs, state_dir):
    paths = [state_dir / "build_state.json", state_dir / "build_progress.npy",
             state_dir / "build_errors.jsonl"]
    for folder, prefix, _, _ in specs:
        for kind in ("vols", "masks", "ids"):
            paths.append(folder / f"{prefix}_{kind}.partial.npy")
            paths.append(folder / f"{prefix}_{kind}.npy")
        paths.append(folder / "corpus_manifest.json")
    return paths


def _open_outputs(ids, specs, resume):
    arrays = []
    uid_dtype = f"<U{max(1, max(map(len, ids)))}"
    for folder, prefix, start, end in specs:
        folder.mkdir(parents=True, exist_ok=True)
        count = end - start
        paths = {
            kind: folder / f"{prefix}_{kind}.partial.npy"
            for kind in ("vols", "masks", "ids")
        }
        partial_exists = [path.is_file() for path in paths.values()]
        if resume and not all(partial_exists):
            raise FileNotFoundError(
                f"Resume state exists but this part is incomplete: {paths}. Use --overwrite."
            )
        mode = "r+" if resume else "w+"
        volumes = np.lib.format.open_memmap(
            paths["vols"], mode=mode, dtype=np.uint8,
            shape=(count, DEPTH, IMG, IMG),
        )
        masks = np.lib.format.open_memmap(
            paths["masks"], mode=mode, dtype=np.uint8, shape=(count, DEPTH),
        )
        id_array = np.lib.format.open_memmap(
            paths["ids"], mode=mode, dtype=uid_dtype, shape=(count,),
        )
        expected_ids = np.asarray(ids[start:end], dtype=uid_dtype)
        if mode == "r+" and not np.array_equal(id_array, expected_ids):
            raise ValueError(f"Partial ID order differs from this build: {paths['ids']}")
        id_array[:] = expected_ids
        id_array.flush()
        arrays.append((start, end, volumes, masks, id_array, paths))
    return arrays


def _write_row(outputs, global_index, volume, mask):
    for start, end, volumes, masks, _, _ in outputs:
        if start <= global_index < end:
            local = global_index - start
            volumes[local] = volume
            masks[local] = mask
            return
    raise IndexError(global_index)


def _flush(outputs):
    for _, _, volumes, masks, ids, _ in outputs:
        volumes.flush()
        masks.flush()
        ids.flush()


def _validate_partial(ids, outputs):
    seen = []
    short_study_count = 0
    minimum_valid_slices = DEPTH
    for start, end, volumes, masks, id_array, _ in outputs:
        expected_shape = (end - start, DEPTH, IMG, IMG)
        if volumes.shape != expected_shape or volumes.dtype != np.uint8:
            raise ValueError(f"Invalid volume array: {volumes.shape}, {volumes.dtype}")
        if masks.shape != expected_shape[:2] or masks.dtype != np.uint8:
            raise ValueError(f"Invalid mask array: {masks.shape}, {masks.dtype}")
        # Validate in small chunks: materializing `volumes != 0` for all 3,200
        # studies at once would require roughly the full 23 GB part size in RAM.
        for local_start in range(0, end - start, 8):
            local_end = min(local_start + 8, end - start)
            mask_chunk = np.asarray(masks[local_start:local_end])
            actual_mask = np.any(
                np.asarray(volumes[local_start:local_end]) != 0, axis=(2, 3)
            ).astype(np.uint8)
            if not np.array_equal(actual_mask, mask_chunk):
                raise ValueError(
                    f"Mask/pixel mismatch in global rows "
                    f"{start + local_start}:{start + local_end}"
                )
            zero_rows = np.where(mask_chunk.sum(axis=1) == 0)[0]
            if len(zero_rows):
                bad = (zero_rows[:5] + start + local_start).tolist()
                raise ValueError(f"All-zero studies found at global rows {bad}")
            valid_counts = mask_chunk.sum(axis=1).astype(int)
            short_study_count += int(np.sum(valid_counts < DEPTH))
            minimum_valid_slices = min(minimum_valid_slices, int(valid_counts.min()))
            for row_mask in mask_chunk:
                count = int(row_mask.sum())
                if not (np.all(row_mask[:count] == 1) and np.all(row_mask[count:] == 0)):
                    raise ValueError("Dense valid slices must be packed contiguously before padding.")
        seen.extend(id_array.astype(str).tolist())
    if seen != ids or len(set(seen)) != len(seen):
        raise ValueError("Final corpus ID order or uniqueness check failed.")
    return short_study_count, minimum_valid_slices


def _finalize(outputs, manifest):
    for _, _, volumes, masks, id_array, paths in outputs:
        for array in (volumes, masks, id_array):
            array.flush()
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        for kind, partial in paths.items():
            final = partial.with_name(partial.name.replace(".partial.npy", ".npy"))
            os.replace(partial, final)
        _atomic_json(final.parent / "corpus_manifest.json", manifest)


def main():
    args = parse_args()
    _check_runtime_dependencies()
    started = time.time()
    competition_dir = Path(args.competition_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_ext_dir = Path(args.output_ext_dir).resolve()
    train_csv, series_csv, images_dir = _resolve_inputs(competition_dir)
    ids, id_source = _load_ids(train_csv, args)
    if not ids:
        raise ValueError("No study IDs selected.")
    if args.split >= len(ids) and not args.limit:
        raise ValueError(f"--split {args.split} must be smaller than full study count {len(ids)}.")
    series_by_study = _load_series(series_csv)
    missing_metadata = [uid for uid in ids if uid not in series_by_study]
    if missing_metadata:
        raise ValueError(f"{len(missing_metadata)} studies lack train_series.csv rows, e.g. {missing_metadata[:3]}")

    specs = _part_specs(len(ids), args.split, output_dir, output_ext_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "version": 4,
        "corpus_kind": "maxspan_dense96_capacity_aware_unique_first_series",
        "series_selection": "first_matching_csv_order_v1",
        "geometry_source": "public_raptor_64_slice_proportions_scaled_to_dense96",
        "geometry_note": (
            "Nominal 27/21/18/12/18 proportions are 1.5x the public Raptor "
            "18/14/12/8/12 layout. Per-study quotas are capacity-capped and "
            "redistributed to other available slots."
        ),
        "competition_dir": str(competition_dir),
        "images_dir": str(images_dir),
        "study_count": len(ids),
        "split": min(args.split, len(ids)),
        "id_source": id_source,
        "img": IMG,
        "crop_mm": CROP_MM,
        "span": [SPAN_LO, SPAN_HI],
        "slots": [list(slot) for slot in SLOTS],
        "depth": DEPTH,
        "allocation": "largest_remainder_capacity_redistribution",
        "unique_source_slices": True,
        "packing": "valid_planes_first_then_masked_padding",
        "intensity_percentiles": [2.0, 98.0],
    }
    state_path = output_dir / "build_state.json"
    progress_path = output_dir / "build_progress.npy"
    errors_path = output_dir / "build_errors.jsonl"

    if args.overwrite:
        for path in _generated_paths(specs, output_dir):
            if path.is_file():
                path.unlink()
    elif any((folder / f"{prefix}_vols.npy").is_file() for folder, prefix, _, _ in specs):
        raise FileExistsError("Final corpus already exists. Use --overwrite to rebuild it.")

    resume = args.resume and state_path.is_file() and progress_path.is_file()
    if resume:
        with open(state_path, encoding="utf-8") as handle:
            old_config = json.load(handle)
        if old_config != config:
            raise ValueError("Existing partial build has different configuration; use --overwrite.")
        done = np.load(progress_path, allow_pickle=False)
        if done.shape != (len(ids),) or done.dtype != np.bool_:
            raise ValueError("Invalid build_progress.npy; use --overwrite.")
    else:
        partial_exists = any(
            (folder / f"{prefix}_vols.partial.npy").is_file()
            for folder, prefix, _, _ in specs
        )
        if partial_exists:
            raise FileExistsError("Partial outputs exist without compatible progress; use --overwrite.")
        done = np.zeros(len(ids), dtype=np.bool_)
        _atomic_json(state_path, config)
        _atomic_npy(progress_path, done)

    outputs = _open_outputs(ids, specs, resume=resume)
    remaining = np.where(~done)[0].tolist()
    total_bytes = len(ids) * DEPTH * IMG * IMG
    print(f"[config] studies={len(ids)} shape=({len(ids)},{DEPTH},{IMG},{IMG}) "
          f"uint8 volume_bytes={total_bytes / 1e9:.2f}GB", flush=True)
    print(f"[config] slots={SLOTS} span={SPAN_LO}:{SPAN_HI} crop={CROP_MM}mm img={IMG}", flush=True)
    print(f"[paths] DICOM={images_dir} output={output_dir} ext={output_ext_dir}", flush=True)
    print(f"[resume] completed={int(done.sum())} remaining={len(remaining)} workers={args.workers}", flush=True)

    pending = []
    failures = []
    tasks = [(index, ids[index]) for index in remaining]
    try:
        with futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(series_by_study, str(images_dir)),
        ) as executor:
            for result in executor.map(_build_study, tasks, chunksize=1):
                index, uid, volume, mask, stats, error = result
                if error is not None:
                    failures.append({"index": index, "StudyInstanceUID": uid, "error": error})
                    print(f"[error] row={index} uid={uid}: {error.splitlines()[0]}", flush=True)
                    continue
                _write_row(outputs, index, volume, mask)
                pending.append(index)
                if len(pending) >= args.flush_every:
                    _flush(outputs)
                    done[pending] = True
                    _atomic_npy(progress_path, done)
                    pending.clear()
                complete = int(done.sum()) + len(pending)
                if complete % 25 == 0 or complete == len(ids):
                    elapsed = max(time.time() - started, 1e-6)
                    rate = complete / elapsed
                    eta = (len(ids) - complete) / max(rate, 1e-9)
                    print(f"[build] {complete}/{len(ids)} | {rate:.2f} studies/s | ETA {eta/60:.1f} min",
                          flush=True)
    finally:
        if pending:
            _flush(outputs)
            done[pending] = True
            _atomic_npy(progress_path, done)

    if failures:
        with open(errors_path, "w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        raise RuntimeError(
            f"{len(failures)} studies failed. Details: {errors_path}. "
            "Fix the decoder/data issue and rerun the same command to resume."
        )
    if not done.all():
        raise RuntimeError(f"Build incomplete: {int(done.sum())}/{len(done)} studies completed.")

    print("[validate] checking shapes, IDs, masks and all-zero studies...", flush=True)
    _flush(outputs)
    short_studies, minimum_valid_slices = _validate_partial(ids, outputs)
    manifest = dict(config)
    manifest.update({
        "completed": True,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": round(time.time() - started, 3),
        "short_study_count": short_studies,
        "minimum_valid_slices": minimum_valid_slices,
        "parts": [
            {"prefix": prefix, "start": start, "end": end, "count": end - start,
             "shape": [end - start, DEPTH, IMG, IMG]}
            for _, prefix, start, end in specs
        ],
    })
    _finalize(outputs, manifest)
    if progress_path.exists():
        progress_path.unlink()
    if errors_path.exists():
        errors_path.unlink()
    print(f"[done] corpus finalized in {(time.time() - started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.", file=sys.stderr, flush=True)
        raise SystemExit(130)
