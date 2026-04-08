from __future__ import annotations

import configparser
import csv
import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import xarray as xr
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from scipy.io import loadmat

from .config import STATIC_METADATA, TASK_DESCRIPTIONS


@dataclass
class SessionMetadata:
    subject: str
    date: str
    session_dir: Path
    ini_path: Path
    ystack_path: Path
    block_to_task: dict[int, str]
    block_to_slice_index: dict[int, int]
    slice_positions_mm: list[float]
    ystack_x_axis_mm: np.ndarray
    ystack_depth_axis_mm: np.ndarray
    plane_wave_angles_deg: list[float] | None
    transmit_frequency_hz: float | None
    compound_sampling_frequency_hz: float | None
    probe_voltage_v: float | None
    depth_mm: tuple[float, float] | None


@dataclass
class RunPlan:
    subject: str
    date: str
    block: int
    task: str
    task_description: str
    slice_index: int
    run_number: int | None
    slice_position_mm: float
    source_hdf: Path
    reference_nifti: Path
    output_nifti: Path


@dataclass
class ConversionSummary:
    planned_runs: int
    converted_runs: int
    skipped_runs: int
    dry_run: bool
    manifest_path: Path


def _parse_number_list(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.replace("\n", "").split(","):
        stripped = token.strip()
        if not stripped:
            continue
        if stripped.lower() == "nan":
            values.append(math.nan)
        else:
            values.append(float(stripped))
    return values


def _median_step(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    diffs = np.diff(values.astype(float))
    return float(np.median(np.abs(diffs)))


def _find_single_file(paths: list[Path], label: str, context: Path) -> Path:
    if len(paths) != 1:
        raise ValueError(
            f"Expected exactly one {label} in {context}, found {len(paths)}."
        )
    return paths[0]


def _load_session_metadata(session_dir: Path) -> SessionMetadata:
    subject = session_dir.parent.name
    date = session_dir.name

    ini_path = _find_single_file(
        sorted(session_dir.glob("*.ini")),
        label=".ini file",
        context=session_dir,
    )
    ystack_path = _find_single_file(
        sorted(session_dir.glob("[0-9][0-9][0-9][0-9]/*fUSiYStack.mat")),
        label="fUSiYStack .mat file",
        context=session_dir,
    )

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation(),
        allow_no_value=True,
    )
    config.read(ini_path)

    block_numbers = [
        int(v) for v in _parse_number_list(config.get("experiment", "block_numbers"))
    ]
    task_names = [
        v.strip()
        for v in config.get("experiment", "tasks").replace("\n", "").split(",")
        if v.strip()
    ]
    task_block_keys = [
        v.strip()
        for v in config.get("experiment", "block_names").replace("\n", "").split(",")
        if v.strip()
    ]

    block_to_task: dict[int, str] = {}
    for task_name, task_block_key in zip(task_names, task_block_keys, strict=False):
        blocks_for_task = [
            int(v)
            for v in _parse_number_list(config.get("experiment", task_block_key))
            if not math.isnan(v)
        ]
        for block in blocks_for_task:
            block_to_task[block] = task_name

    block_to_slice_index: dict[int, int] = {}
    mapping = _parse_number_list(config.get("fusi", "mapping_block2slices"))
    for block in block_numbers:
        mapping_index = block - 1
        if mapping_index >= len(mapping):
            continue
        mapped_slice = mapping[mapping_index]
        if math.isnan(mapped_slice):
            continue
        block_to_slice_index[block] = int(mapped_slice)

    slice_positions_mm = [
        float(v)
        for v in _parse_number_list(config.get("fusi", "slice_positions"))
        if not math.isnan(v)
    ]

    ystack = loadmat(ystack_path, simplify_cells=True)
    params = ystack["params"]
    par_seq = params["parSeq"]
    hq = par_seq["HQ"]
    doppler = ystack["Doppler"]

    x_axis = np.asarray(doppler["xAxis"], dtype=float).ravel()
    depth_axis = np.asarray(doppler["zAxis"], dtype=float).ravel()

    depth_mm: tuple[float, float] | None
    if depth_axis.size > 0:
        depth_mm = (float(np.min(depth_axis)), float(np.max(depth_axis)))
    else:
        depth_mm = None

    plane_wave_angles = np.asarray(hq.get("angles", []), dtype=float).ravel()
    plane_wave_angles_deg = (
        [float(v) for v in plane_wave_angles.tolist()]
        if plane_wave_angles.size > 0
        else None
    )

    transmit_frequency_hz = (
        float(par_seq["TF"]) * 1e6 if "TF" in par_seq and par_seq["TF"] else None
    )
    compound_sampling_frequency_hz = (
        float(hq["Frate"]) if "Frate" in hq and hq["Frate"] else None
    )
    probe_voltage_v = (
        float(par_seq["HVset"])
        if "HVset" in par_seq and par_seq["HVset"] is not None
        else None
    )

    return SessionMetadata(
        subject=subject,
        date=date,
        session_dir=session_dir,
        ini_path=ini_path,
        ystack_path=ystack_path,
        block_to_task=block_to_task,
        block_to_slice_index=block_to_slice_index,
        slice_positions_mm=slice_positions_mm,
        ystack_x_axis_mm=x_axis,
        ystack_depth_axis_mm=depth_axis,
        plane_wave_angles_deg=plane_wave_angles_deg,
        transmit_frequency_hz=transmit_frequency_hz,
        compound_sampling_frequency_hz=compound_sampling_frequency_hz,
        probe_voltage_v=probe_voltage_v,
        depth_mm=depth_mm,
    )


def _load_reference_axes(reference_nifti: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import confusius as cf
    except ImportError as exc:
        raise RuntimeError(
            "confusius is required for conversion. Install it in this environment, "
            "or run with `uv run --with ../confusius ...`."
        ) from exc

    reference = cf.load(reference_nifti)
    x_values = np.asarray(reference.coords["x"].values, dtype=float)
    depth_values = np.asarray(reference.coords["z"].values, dtype=float)
    return x_values, depth_values


def _repair_times(
    times: np.ndarray,
    n_frames: int,
) -> tuple[np.ndarray, int, str]:
    n_times = times.size
    if n_times == n_frames:
        return times, n_frames, "none"

    if n_times == n_frames - 1:
        return times, n_times, "drop_last_frame"

    raise ValueError(
        f"Unexpected mismatch between data frames ({n_frames}) and time samples ({n_times})."
    )


def _resolve_slice_position(slice_positions_mm: list[float], slice_index: int) -> float:
    if not slice_positions_mm:
        return 0.0
    if 0 <= slice_index < len(slice_positions_mm):
        return float(slice_positions_mm[slice_index])
    return float(slice_positions_mm[0])


def _collect_run_plans(
    src_subjects_dir: Path,
    out_dir: Path,
    subjects_filter: set[str] | None,
) -> tuple[list[RunPlan], dict[tuple[str, str], SessionMetadata]]:
    session_dirs = sorted(
        p for p in src_subjects_dir.glob("*/*") if p.is_dir() and any(p.glob("*.ini"))
    )

    session_metadata: dict[tuple[str, str], SessionMetadata] = {}
    raw_runs: list[tuple[SessionMetadata, Path, int, str, int, float, Path]] = []

    for session_dir in session_dirs:
        metadata = _load_session_metadata(session_dir)
        if subjects_filter is not None and metadata.subject not in subjects_filter:
            continue

        session_metadata[(metadata.subject, metadata.date)] = metadata

        fusi_files = sorted(
            session_dir.glob("*/fusi/*svddrop015.hdf"),
            key=lambda p: int(p.parent.parent.name),
        )
        for source_hdf in fusi_files:
            block = int(source_hdf.parent.parent.name)
            task = metadata.block_to_task.get(block)
            if task is None:
                continue
            slice_index = metadata.block_to_slice_index.get(block)
            if slice_index is None:
                continue

            slice_position_mm = _resolve_slice_position(
                metadata.slice_positions_mm,
                slice_index,
            )

            reference_nifti = (
                session_dir
                / "allenccf_align"
                / f"{metadata.date}_{metadata.subject}_fusi_slice_acq.nii.gz"
            )
            if not reference_nifti.exists():
                raise FileNotFoundError(
                    f"Missing reference NIfTI for {metadata.subject} {metadata.date}: {reference_nifti}"
                )

            raw_runs.append(
                (
                    metadata,
                    source_hdf,
                    block,
                    task,
                    slice_index,
                    slice_position_mm,
                    reference_nifti,
                )
            )

    raw_runs.sort(key=lambda item: (item[0].subject, item[0].date, item[2]))

    key_counts: defaultdict[tuple[str, str, str, int], int] = defaultdict(int)
    for (
        metadata,
        _source_hdf,
        _block,
        task,
        slice_index,
        _slice_position,
        _ref,
    ) in raw_runs:
        key_counts[(metadata.subject, metadata.date, task, slice_index)] += 1

    key_run_counters: defaultdict[tuple[str, str, str, int], int] = defaultdict(int)
    plans: list[RunPlan] = []
    for (
        metadata,
        source_hdf,
        block,
        task,
        slice_index,
        slice_position_mm,
        reference_nifti,
    ) in raw_runs:
        run_key = (metadata.subject, metadata.date, task, slice_index)
        run_number: int | None = None
        if key_counts[run_key] > 1:
            key_run_counters[run_key] += 1
            run_number = key_run_counters[run_key]

        ses = metadata.date.replace("-", "")
        run_entity = f"_run-{run_number:02d}" if run_number is not None else ""
        output_name = (
            f"sub-{metadata.subject}"
            f"_ses-{ses}"
            f"_task-{task}"
            f"_acq-slice{slice_index + 1:02d}"
            f"{run_entity}"
            "_pwd.nii.gz"
        )
        output_nifti = (
            out_dir / f"sub-{metadata.subject}" / f"ses-{ses}" / "fusi" / output_name
        )

        plans.append(
            RunPlan(
                subject=metadata.subject,
                date=metadata.date,
                block=block,
                task=task,
                task_description=TASK_DESCRIPTIONS.get(
                    task,
                    f"{task} task as defined in the original experiment logs.",
                ),
                slice_index=slice_index,
                run_number=run_number,
                slice_position_mm=slice_position_mm,
                source_hdf=source_hdf,
                reference_nifti=reference_nifti,
                output_nifti=output_nifti,
            )
        )

    return plans, session_metadata


def _build_dataset_metadata(out_dir: Path, subjects: list[str]) -> None:
    dataset_description = {
        "Name": (
            "Simultaneous functional ultrasound and electrophysiology recordings "
            "of neural activity in awake mice"
        ),
        "BIDSVersion": "1.11.1",
        "DatasetType": "raw",
        "License": "CC-BY-4.0",
        "Authors": [
            "A. O. Nunez-Elizalde",
            "M. Krumin",
            "C. B. Reddy",
            "G. Montaldo",
            "A. Urban",
            "K. D. Harris",
            "M. Carandini",
        ],
        "ReferencesAndLinks": [
            "doi:10.1016/j.neuron.2022.02.012",
            "https://github.com/anwarnunez/fusi",
        ],
        "SourceDatasets": [
            {
                "DOI": "doi:10.6084/m9.figshare.19316228",
                "URL": "https://figshare.com/articles/dataset/19316228",
            }
        ],
        "GeneratedBy": [
            {
                "Name": "nunez-elizalde-2022-bids",
                "Description": "Conversion scripts for fUSI-only BIDS export.",
            }
        ],
    }
    (out_dir / "dataset_description.json").write_text(
        json.dumps(dataset_description, indent=2) + "\n"
    )

    readme_text = (
        "This dataset contains simultaneous recordings of neural activity with "
        "electrodes and blood flow with functional ultrasound in awake mice.\n\n"
        "It is a conversion to fUSI-BIDS of the original dataset released at:\n"
        "doi:10.6084/m9.figshare.19316228\n\n"
        "These data appear in the article:\n"
        "Neural correlates of blood flow measured by ultrasound. "
        "Nunez-Elizalde AO, Krumin M, Reddy CB, Montaldo G, Urban A, "
        "Harris KD, and Carandini M. Neuron (2022).\n"
        "doi:10.1016/j.neuron.2022.02.012\n\n"
        "Software for the original analysis of these data:\n"
        "https://github.com/anwarnunez/fusi\n"
    )
    (out_dir / "README").write_text(readme_text)

    bidsignore_text = (
        "dataset_index.json\n"
        "code/**\n"
        "sourcedata/**\n"
        "derivatives/**\n"
        "sub-*/ses-*/fusi\n"
        "sub-*/ses-*/fusi/**\n"
        "sub-*/ses-*/angio\n"
        "sub-*/ses-*/angio/**\n"
    )
    (out_dir / ".bidsignore").write_text(bidsignore_text)

    participants_tsv = out_dir / "participants.tsv"
    with participants_tsv.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["participant_id", "species"])
        for subject in sorted(subjects):
            writer.writerow([f"sub-{subject}", "mus musculus"])


def _session_label(date: str) -> str:
    return date.replace("-", "")


def _angio_output_path(out_dir: Path, subject: str, date: str) -> Path:
    ses_label = _session_label(date)
    filename = f"sub-{subject}_ses-{ses_label}_pwd.nii.gz"
    return out_dir / f"sub-{subject}" / f"ses-{ses_label}" / "angio" / filename


def _find_angio_source(metadata: SessionMetadata) -> Path | None:
    align_dir = metadata.session_dir / "allenccf_align"
    if not align_dir.exists():
        return None

    candidates = [
        align_dir / f"{metadata.date}_{metadata.subject}_fusi_ystack_sqrt.nii.gz",
        align_dir / f"{metadata.date}_{metadata.subject}_fusi_ystack.nii.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _build_angio_sidecar(
    metadata: SessionMetadata, *, source_file: Path
) -> dict[str, Any]:
    sidecar: dict[str, Any] = {
        "Description": "3D fUSI angiography volume from session y-stack.",
        "Manufacturer": STATIC_METADATA["manufacturer"],
        "ManufacturersModelName": STATIC_METADATA["manufacturers_model_name"],
        "SoftwareVersion": STATIC_METADATA["software_version"],
        "ProbeManufacturer": STATIC_METADATA["probe_manufacturer"],
        "ProbeType": STATIC_METADATA["probe_type"],
        "ProbeModel": STATIC_METADATA["probe_model"],
        "ProbeCentralFrequency": STATIC_METADATA["probe_central_frequency"],
        "ProbeNumberOfElements": STATIC_METADATA["probe_number_of_elements"],
        "ProbePitch": STATIC_METADATA["probe_pitch"],
        "ProbeFocalWidth": STATIC_METADATA["probe_focal_width"],
        "ProbeFocalDepth": STATIC_METADATA["probe_focal_depth"],
        "SourceFile": source_file.name,
    }

    if metadata.depth_mm is not None:
        sidecar["Depth"] = [float(metadata.depth_mm[0]), float(metadata.depth_mm[1])]
    if metadata.transmit_frequency_hz is not None:
        sidecar["UltrasoundTransmitFrequency"] = metadata.transmit_frequency_hz
    if metadata.compound_sampling_frequency_hz is not None:
        sidecar["CompoundSamplingFrequency"] = metadata.compound_sampling_frequency_hz
    if metadata.plane_wave_angles_deg is not None:
        sidecar["PlaneWaveAngles"] = metadata.plane_wave_angles_deg
    if metadata.probe_voltage_v is not None:
        sidecar["ProbeVoltage"] = metadata.probe_voltage_v

    return sidecar


def _probe_entity_from_name(filename: str) -> str:
    match = re.search(r"probe(\d+)", filename)
    if match is None:
        return "probe"
    return f"probe{match.group(1)}"


def _derivative_filename(subject: str, date: str, source_name: str) -> str:
    ses_label = _session_label(date)
    prefix = f"sub-{subject}_ses-{ses_label}"

    if source_name.endswith("_fusi_slice_acq.nii.gz"):
        return f"{prefix}_desc-sliceacq_pwd.nii.gz"

    if source_name.endswith("_alleccf_atlas_resampled_fusi_scaled01x_byindex.nii.gz"):
        return f"{prefix}_space-fusi_desc-allenccf_dseg.nii.gz"

    if source_name.endswith("_fusi_probe00_track.nii.gz"):
        return f"{prefix}_desc-probe00_track.nii.gz"

    if source_name.endswith("_fusi_probe01_track.nii.gz"):
        return f"{prefix}_desc-probe01_track.nii.gz"

    if source_name.endswith("_estimated_probe00_3Dtrack_manual.nii.gz"):
        return f"{prefix}_desc-probe00manual_track.nii.gz"

    if source_name.endswith("_estimated_probe01_3Dtrack_manual.nii.gz"):
        return f"{prefix}_desc-probe01manual_track.nii.gz"

    if source_name.endswith("_estimated_probe00_3Dtrack_manual_mask.nii.gz"):
        return f"{prefix}_desc-probe00manual_mask.nii.gz"

    if source_name.endswith("_estimated_probe01_3Dtrack_manual_mask.nii.gz"):
        return f"{prefix}_desc-probe01manual_mask.nii.gz"

    return source_name


def _to_confusius_stack_convention(da: xr.DataArray) -> xr.DataArray:
    if tuple(da.dims) != ("z", "y", "x"):
        return da

    if da.sizes["z"] <= da.sizes["y"]:
        return da

    data = np.asarray(da.data).transpose(1, 0, 2)
    z_coord = xr.DataArray(
        np.asarray(da.coords["y"].values),
        dims=("z",),
        attrs=dict(da.coords["y"].attrs),
    )
    y_coord = xr.DataArray(
        np.asarray(da.coords["z"].values),
        dims=("y",),
        attrs=dict(da.coords["z"].attrs),
    )
    x_coord = xr.DataArray(
        np.asarray(da.coords["x"].values),
        dims=("x",),
        attrs=dict(da.coords["x"].attrs),
    )
    attrs = dict(da.attrs)
    attrs.pop("affines", None)
    attrs.pop("qform_code", None)
    attrs.pop("sform_code", None)

    return xr.DataArray(
        data,
        dims=("z", "y", "x"),
        coords={"z": z_coord, "y": y_coord, "x": x_coord},
        attrs=attrs,
        name=da.name,
    )


def _save_conformed_nifti(source: Path, destination: Path) -> None:
    try:
        import confusius as cf
    except ImportError as exc:
        raise RuntimeError(
            "confusius is required for conversion. Install it in this environment, "
            "or run with `uv run --with ../confusius ...`."
        ) from exc

    da = cf.load(source)
    da_conformed = _to_confusius_stack_convention(da)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cf.save(da_conformed, destination)


def _write_derivatives_dataset_description(derivatives_root: Path) -> None:
    description = {
        "Name": "Allen CCF alignment outputs",
        "BIDSVersion": "1.11.1",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "allenccf_align",
                "Description": "Copies allenccf_align outputs into BIDS derivatives.",
            }
        ],
        "SourceDatasets": [
            {
                "DOI": "doi:10.6084/m9.figshare.19316228",
                "URL": "https://figshare.com/articles/dataset/19316228",
            },
        ],
    }
    (derivatives_root / "dataset_description.json").write_text(
        json.dumps(description, indent=2) + "\n"
    )


def _write_sourcedata_readme(sourcedata_root: Path) -> None:
    readme = (
        "This folder stores source alignment artifacts that are not converted into "
        "BIDS derivative images.\n"
        "Currently it contains original probe-track HDF files copied from the source "
        "dataset.\n"
    )
    (sourcedata_root / "README").write_text(readme)


def _extract_probe_track_hdf(
    source_hdf: Path,
    destination_tsv: Path,
    destination_json: Path,
) -> None:
    with h5py.File(source_hdf, "r") as handle:
        track_ijk = np.asarray(handle.get("probe_track_ijk_voxels", []))
        track_xyz = np.asarray(handle.get("probe_track_xyzmm", []), dtype=float)
        probe_depth = np.asarray(handle.get("probe_depth", []), dtype=float).ravel()
        insertion_voxel = np.asarray(handle.get("probe_insertion_voxel", [])).ravel()
        probe_tip_mm = np.asarray(handle.get("probe_tip_mm", []), dtype=float).ravel()

    n_candidates = []
    if track_ijk.ndim == 2:
        n_candidates.append(track_ijk.shape[0])
    if track_xyz.ndim == 2:
        n_candidates.append(track_xyz.shape[0])
    if probe_depth.ndim == 1:
        n_candidates.append(probe_depth.size)
    n_points = min(n_candidates) if n_candidates else 0

    fieldnames = [
        "point_index",
        "voxel_i",
        "voxel_j",
        "voxel_k",
        "x_mm",
        "y_mm",
        "z_mm",
        "depth_mm",
    ]
    with destination_tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for index in range(n_points):
            row: dict[str, Any] = {"point_index": index}
            if track_ijk.ndim == 2 and track_ijk.shape[1] >= 3:
                row["voxel_i"] = int(track_ijk[index, 0])
                row["voxel_j"] = int(track_ijk[index, 1])
                row["voxel_k"] = int(track_ijk[index, 2])
            if track_xyz.ndim == 2 and track_xyz.shape[1] >= 3:
                row["x_mm"] = float(track_xyz[index, 0])
                row["y_mm"] = float(track_xyz[index, 1])
                row["z_mm"] = float(track_xyz[index, 2])
            if probe_depth.ndim == 1 and probe_depth.size > index:
                row["depth_mm"] = float(probe_depth[index])
            writer.writerow(row)

    sidecar = {
        "Description": "Probe trajectory sampled points extracted from source HDF.",
        "SourceFile": source_hdf.name,
        "point_index": {
            "Description": "Point index along the probe trajectory.",
        },
        "voxel_i": {
            "Description": "Probe track voxel index i.",
        },
        "voxel_j": {
            "Description": "Probe track voxel index j.",
        },
        "voxel_k": {
            "Description": "Probe track voxel index k.",
        },
        "x_mm": {
            "Description": "Probe point x position in millimeters.",
            "Units": "mm",
        },
        "y_mm": {
            "Description": "Probe point y position in millimeters.",
            "Units": "mm",
        },
        "z_mm": {
            "Description": "Probe point z position in millimeters.",
            "Units": "mm",
        },
        "depth_mm": {
            "Description": "Depth value associated with each track point.",
            "Units": "mm",
        },
    }
    if insertion_voxel.size == 3:
        sidecar["ProbeInsertionVoxel"] = [
            int(insertion_voxel[0]),
            int(insertion_voxel[1]),
            int(insertion_voxel[2]),
        ]
    if probe_tip_mm.size == 3:
        sidecar["ProbeTipMm"] = [
            float(probe_tip_mm[0]),
            float(probe_tip_mm[1]),
            float(probe_tip_mm[2]),
        ]
    destination_json.write_text(json.dumps(sidecar, indent=2) + "\n")


def _copy_angio_and_derivatives(
    out_dir: Path,
    plans: list[RunPlan],
    session_metadata: dict[tuple[str, str], SessionMetadata],
) -> None:
    sessions = {(plan.subject, plan.date) for plan in plans}

    derivatives_root = out_dir / "derivatives" / "allenccf_align"
    derivatives_root.mkdir(parents=True, exist_ok=True)
    _write_derivatives_dataset_description(derivatives_root)

    sourcedata_root = out_dir / "sourcedata" / "allenccf_align"
    sourcedata_root.mkdir(parents=True, exist_ok=True)
    _write_sourcedata_readme(sourcedata_root)

    derivative_patterns = [
        "*_fusi_slice_acq.nii.gz",
        "*_alleccf_atlas_resampled_fusi_scaled01x_byindex.nii.gz",
        "*_fusi_probe*_track.nii.gz",
        "*_estimated_probe*_3Dtrack_manual.nii.gz",
        "*_estimated_probe*_3Dtrack_manual_mask.nii.gz",
    ]

    for key in sorted(sessions):
        metadata = session_metadata[key]
        ses_label = _session_label(metadata.date)

        angio_source = _find_angio_source(metadata)
        if angio_source is not None:
            angio_output = _angio_output_path(out_dir, metadata.subject, metadata.date)
            _save_conformed_nifti(angio_source, angio_output)

            angio_json = angio_output.with_suffix("").with_suffix(".json")
            angio_sidecar = _build_angio_sidecar(metadata, source_file=angio_source)
            angio_json.write_text(json.dumps(angio_sidecar, indent=2) + "\n")

        align_dir = metadata.session_dir / "allenccf_align"
        if not align_dir.exists():
            continue

        destination = (
            derivatives_root / f"sub-{metadata.subject}" / f"ses-{ses_label}" / "fusi"
        )
        destination.mkdir(parents=True, exist_ok=True)
        sourcedata_destination = (
            sourcedata_root / f"sub-{metadata.subject}" / f"ses-{ses_label}" / "fusi"
        )
        sourcedata_destination.mkdir(parents=True, exist_ok=True)

        for pattern in derivative_patterns:
            for source in sorted(align_dir.glob(pattern)):
                target_name = _derivative_filename(
                    metadata.subject,
                    metadata.date,
                    source.name,
                )
                _save_conformed_nifti(source, destination / target_name)

        for source_hdf in sorted(
            align_dir.glob("*_estimated_probe*_3Dtrack_manual.hdf")
        ):
            shutil.copy2(source_hdf, sourcedata_destination / source_hdf.name)
            probe_entity = _probe_entity_from_name(source_hdf.name)
            ses_label = _session_label(metadata.date)
            track_base = f"sub-{metadata.subject}_ses-{ses_label}_desc-{probe_entity}manual_track"
            extracted_tsv = destination / f"{track_base}.tsv"
            extracted_json = destination / f"{track_base}.json"
            _extract_probe_track_hdf(source_hdf, extracted_tsv, extracted_json)


def _timeline_path_for_run(plan: RunPlan) -> Path:
    block_dir = plan.source_hdf.parent.parent
    return block_dir / f"{plan.date}_{plan.block}_{plan.subject}_Timeline.mat"


def _session_acq_time_iso(runs: list[RunPlan]) -> str:
    if not runs:
        return "n/a"

    first_run = min(runs, key=lambda run: run.block)
    timeline_path = _timeline_path_for_run(first_run)
    if not timeline_path.exists():
        return "n/a"

    try:
        timeline = loadmat(timeline_path, simplify_cells=True).get("Timeline", {})
        start_str = timeline.get("startDateTimeStr")
        if not isinstance(start_str, str):
            return "n/a"
        start_dt = datetime.strptime(start_str, "%d-%b-%Y %H:%M:%S")

        with h5py.File(first_run.source_hdf, "r") as h5:
            times = np.asarray(h5["times"], dtype=np.float64)
        if times.size == 0:
            return "n/a"

        acq_dt = start_dt + timedelta(seconds=float(times[0]))
        return acq_dt.isoformat(timespec="milliseconds")
    except Exception:
        return "n/a"


def _parse_udp_event(raw_event: Any) -> list[str]:
    if isinstance(raw_event, str):
        return raw_event.split()
    return []


def _extract_stimulus_events(
    timeline_path: Path,
    *,
    first_volume_time: float,
    task_name: str,
) -> list[dict[str, Any]]:
    timeline = loadmat(timeline_path, simplify_cells=True).get("Timeline", {})
    if not isinstance(timeline, dict):
        return []

    raw_events = timeline.get("mpepUDPEvents", [])
    raw_times = timeline.get("mpepUDPTimes", [])

    if isinstance(raw_events, str):
        events = [raw_events]
    else:
        events = np.asarray(raw_events, dtype=object).ravel().tolist()
    times = np.asarray(raw_times, dtype=float).ravel().tolist()

    active_starts: defaultdict[tuple[str, str], list[tuple[float, str, str]]] = (
        defaultdict(list)
    )
    rows: list[dict[str, Any]] = []

    for raw_event, raw_time in zip(events, times, strict=False):
        tokens = _parse_udp_event(raw_event)
        if not tokens:
            continue

        tag = tokens[0]
        if tag == "StimStart" and len(tokens) >= 7:
            block_repeat = tokens[4]
            stimulus_id = tokens[5]
            stimulus_duration_code = tokens[6]
            key = (block_repeat, stimulus_id)
            active_starts[key].append(
                (float(raw_time), stimulus_duration_code, str(raw_event))
            )
            continue

        if tag == "StimEnd" and len(tokens) >= 6:
            block_repeat = tokens[4]
            stimulus_id = tokens[5]
            key = (block_repeat, stimulus_id)
            if not active_starts[key]:
                continue
            onset_abs, stimulus_duration_code, source_event = active_starts[key].pop(0)
            duration = float(raw_time) - onset_abs
            if duration <= 0:
                continue

            rows.append(
                {
                    "onset": onset_abs - first_volume_time,
                    "duration": duration,
                    "trial_type": task_name,
                    "block_repeat": block_repeat,
                    "stimulus_id": stimulus_id,
                    "stimulus_duration_code": stimulus_duration_code,
                    "source_event": source_event,
                }
            )

    rows.sort(key=lambda row: float(row["onset"]))
    return rows


def _event_paths(output_nifti: Path) -> tuple[Path, Path]:
    stem = output_nifti.name.removesuffix("_pwd.nii.gz")
    events_tsv = output_nifti.parent / f"{stem}_events.tsv"
    events_json = output_nifti.parent / f"{stem}_events.json"
    return events_tsv, events_json


def _write_events_files(
    events_tsv: Path, events_json: Path, rows: list[dict[str, Any]]
) -> None:
    fieldnames = [
        "onset",
        "duration",
        "trial_type",
        "block_repeat",
        "stimulus_id",
        "stimulus_duration_code",
        "source_event",
    ]
    with events_tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    events_sidecar = {
        "onset": {
            "Description": "Event onset relative to the first acquired fUSI volume.",
            "Units": "s",
        },
        "duration": {
            "Description": "Event duration from matched StimStart/StimEnd UDP events.",
            "Units": "s",
        },
        "trial_type": {
            "Description": "Task label from session metadata.",
        },
        "block_repeat": {
            "Description": "Within-block repeat index from stimulus UDP event payload.",
        },
        "stimulus_id": {
            "Description": "Stimulus identity code from stimulus UDP event payload.",
        },
        "stimulus_duration_code": {
            "Description": (
                "Raw duration-like code from StimStart payload (e.g., 20 or 499)."
            ),
        },
        "source_event": {
            "Description": "Original StimStart UDP event string for provenance.",
        },
    }
    events_json.write_text(json.dumps(events_sidecar, indent=2) + "\n")


def _write_bids_tabular_metadata(
    out_dir: Path,
    plans: list[RunPlan],
) -> None:
    subjects = sorted({plan.subject for plan in plans})
    _build_dataset_metadata(out_dir, subjects)

    participants_json = {
        "participant_id": {
            "Description": "Unique participant identifier.",
        },
        "species": {
            "Description": "Binomial species name.",
            "Levels": {"mus musculus": "House mouse."},
        },
    }
    (out_dir / "participants.json").write_text(
        json.dumps(participants_json, indent=2) + "\n"
    )

    runs_by_subject_session: defaultdict[tuple[str, str], list[RunPlan]] = defaultdict(
        list
    )
    for plan in plans:
        runs_by_subject_session[(plan.subject, plan.date)].append(plan)

    sessions_by_subject: defaultdict[str, list[str]] = defaultdict(list)
    for subject, date in runs_by_subject_session:
        sessions_by_subject[subject].append(date)

    for subject, dates in sessions_by_subject.items():
        subject_dir = out_dir / f"sub-{subject}"
        subject_dir.mkdir(parents=True, exist_ok=True)

        unique_dates = sorted(set(dates))
        if len(unique_dates) > 1:
            sessions_tsv = subject_dir / f"sub-{subject}_sessions.tsv"
            with sessions_tsv.open("w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["session_id", "acq_time"])
                for date in unique_dates:
                    ses_label = date.replace("-", "")
                    runs = sorted(
                        runs_by_subject_session[(subject, date)],
                        key=lambda run: run.block,
                    )
                    writer.writerow([f"ses-{ses_label}", _session_acq_time_iso(runs)])

            sessions_json = {
                "session_id": {
                    "Description": "Session identifier matching BIDS `ses-<label>`.",
                },
                "acq_time": {
                    "Description": (
                        "Acquisition time of the first data point of the first run "
                        "in this session."
                    ),
                },
            }
            (subject_dir / f"sub-{subject}_sessions.json").write_text(
                json.dumps(sessions_json, indent=2) + "\n"
            )

        for date in unique_dates:
            ses_label = date.replace("-", "")
            session_dir = subject_dir / f"ses-{ses_label}"
            session_dir.mkdir(parents=True, exist_ok=True)


def _convert_run(
    plan: RunPlan,
    *,
    metadata: SessionMetadata,
    overwrite: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "subject": plan.subject,
        "date": plan.date,
        "block": plan.block,
        "task": plan.task,
        "slice_index": plan.slice_index,
        "source_hdf": str(plan.source_hdf),
        "output_nifti": str(plan.output_nifti),
    }

    if plan.output_nifti.exists() and not overwrite:
        result["status"] = "skipped_exists"
        return result

    with h5py.File(plan.source_hdf, "r") as h5:
        data = np.asarray(h5["data"], dtype=np.float32)
        times = np.asarray(h5["times"], dtype=np.float64)

    repaired_times, n_target_frames, time_repair = _repair_times(
        times,
        n_frames=data.shape[0],
    )
    data = data[:n_target_frames, :, :]

    x_reference, depth_reference = _load_reference_axes(plan.reference_nifti)
    x_values = (
        x_reference
        if x_reference.size == data.shape[2]
        else metadata.ystack_x_axis_mm[: data.shape[2]]
    )
    depth_values = (
        depth_reference
        if depth_reference.size == data.shape[1]
        else metadata.ystack_depth_axis_mm[: data.shape[1]]
    )

    if x_values.size != data.shape[2] or depth_values.size != data.shape[1]:
        raise ValueError(
            f"Coordinate length mismatch for {plan.source_hdf}: "
            f"data shape={data.shape}, x={x_values.size}, depth={depth_values.size}."
        )

    x_step = _median_step(x_values)
    depth_step = _median_step(depth_values)
    slice_step = _median_step(np.asarray(metadata.slice_positions_mm, dtype=float))

    time_coord = xr.DataArray(
        repaired_times,
        dims=("time",),
        attrs={
            "units": "s",
            "volume_acquisition_reference": "start",
            "volume_acquisition_duration": STATIC_METADATA[
                "power_doppler_integration_duration"
            ],
        },
    )

    spatial_coords = {
        "z": xr.DataArray(
            np.asarray([plan.slice_position_mm], dtype=float),
            dims=("z",),
            attrs={
                "units": "mm",
                "voxdim": float(slice_step) if slice_step is not None else 1.0,
            },
        ),
        "y": xr.DataArray(
            depth_values,
            dims=("y",),
            attrs={
                "units": "mm",
                "voxdim": float(depth_step) if depth_step is not None else 1.0,
            },
        ),
        "x": xr.DataArray(
            x_values,
            dims=("x",),
            attrs={
                "units": "mm",
                "voxdim": float(x_step) if x_step is not None else 1.0,
            },
        ),
    }

    attrs = dict(STATIC_METADATA)
    attrs["task_name"] = plan.task
    attrs["task_description"] = plan.task_description

    if metadata.depth_mm is not None:
        attrs["depth"] = [float(metadata.depth_mm[0]), float(metadata.depth_mm[1])]
    if metadata.transmit_frequency_hz is not None:
        attrs["transmit_frequency"] = metadata.transmit_frequency_hz
    if metadata.compound_sampling_frequency_hz is not None:
        attrs["compound_sampling_frequency"] = metadata.compound_sampling_frequency_hz
    if metadata.plane_wave_angles_deg is not None:
        attrs["plane_wave_angles"] = metadata.plane_wave_angles_deg
    if metadata.probe_voltage_v is not None:
        attrs["probe_voltage"] = metadata.probe_voltage_v

    da = xr.DataArray(
        data[:, np.newaxis, :, :],
        dims=("time", "z", "y", "x"),
        coords={"time": time_coord, **spatial_coords},
        attrs=attrs,
        name="pwd",
    )

    try:
        import confusius as cf
    except ImportError as exc:
        raise RuntimeError(
            "confusius is required for conversion. Install it in this environment, "
            "or run with `uv run --with ../confusius ...`."
        ) from exc

    plan.output_nifti.parent.mkdir(parents=True, exist_ok=True)
    cf.save(da, plan.output_nifti)

    events_tsv, events_json = _event_paths(plan.output_nifti)
    timeline_path = _timeline_path_for_run(plan)
    events_rows: list[dict[str, Any]] = []
    if timeline_path.exists():
        events_rows = _extract_stimulus_events(
            timeline_path,
            first_volume_time=float(repaired_times[0]),
            task_name=plan.task,
        )
    if events_rows:
        _write_events_files(events_tsv, events_json, events_rows)
        result["events_tsv"] = str(events_tsv)
        result["n_events"] = len(events_rows)
    else:
        events_tsv.unlink(missing_ok=True)
        events_json.unlink(missing_ok=True)
        result["n_events"] = 0

    result["status"] = "converted"
    result["n_frames"] = int(data.shape[0])
    result["n_times_original"] = int(times.size)
    result["time_repair"] = time_repair
    return result


def _write_manifest(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    manifest_dir = out_dir / "code"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "conversion_manifest.tsv"
    if not rows:
        return manifest_path
    fieldnames = sorted({key for row in rows for key in row})
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return manifest_path


def convert(
    *,
    src: Path,
    out: Path,
    subjects: list[str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> ConversionSummary:
    src_resolved = src.expanduser().resolve()
    out_resolved = out.expanduser().resolve()

    if not src_resolved.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_resolved}")

    subjects_filter = set(subjects) if subjects else None
    plans, session_metadata = _collect_run_plans(
        src_resolved, out_resolved, subjects_filter
    )

    out_resolved.mkdir(parents=True, exist_ok=True)

    if not plans:
        return ConversionSummary(
            planned_runs=0,
            converted_runs=0,
            skipped_runs=0,
            dry_run=dry_run,
            manifest_path=out_resolved / "code" / "conversion_manifest.tsv",
        )

    _write_bids_tabular_metadata(out_resolved, plans)

    rows: list[dict[str, Any]] = []
    if dry_run:
        for plan in plans:
            rows.append(
                {
                    "status": "planned",
                    "subject": plan.subject,
                    "date": plan.date,
                    "block": plan.block,
                    "task": plan.task,
                    "slice_index": plan.slice_index,
                    "source_hdf": str(plan.source_hdf),
                    "output_nifti": str(plan.output_nifti),
                }
            )
        manifest_path = _write_manifest(out_resolved, rows)
        return ConversionSummary(
            planned_runs=len(plans),
            converted_runs=0,
            skipped_runs=0,
            dry_run=True,
            manifest_path=manifest_path,
        )

    _copy_angio_and_derivatives(out_resolved, plans, session_metadata)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    with progress:
        task_id = progress.add_task("Converting runs", total=len(plans))
        for plan in plans:
            progress.update(
                task_id,
                description=(
                    f"Converting {plan.subject} {plan.date} "
                    f"block {plan.block} ({plan.task})"
                ),
            )
            metadata = session_metadata[(plan.subject, plan.date)]
            row = _convert_run(
                plan,
                metadata=metadata,
                overwrite=overwrite,
            )
            rows.append(row)
            progress.advance(task_id)

    manifest_path = _write_manifest(out_resolved, rows)
    converted = sum(1 for row in rows if row.get("status") == "converted")
    skipped = sum(1 for row in rows if row.get("status") == "skipped_exists")
    return ConversionSummary(
        planned_runs=len(plans),
        converted_runs=converted,
        skipped_runs=skipped,
        dry_run=False,
        manifest_path=manifest_path,
    )


__all__ = ["ConversionSummary", "convert"]
