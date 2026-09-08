"""Import metadata-labelled TIFF time series into CellClicker projects."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import tifffile
from PIL import Image

from benchmarks.core import preprocess_image
from .project_paths import CELL_REGIONS_FILENAME


SUPPORTED_AXES = frozenset("TCZYX")


def find_tiff_files(source_directory):
    """Return TIFF files directly contained in ``source_directory`` in name order."""
    source_directory = Path(source_directory)
    if not source_directory.is_dir():
        raise NotADirectoryError(f"TIFF source directory does not exist: {source_directory}")
    files = sorted(path for path in source_directory.iterdir() if path.is_file() and path.suffix.lower() in {".tif", ".tiff"})
    if not files:
        raise FileNotFoundError(f"No .tif or .tiff files found in {source_directory}")
    return files


def _read_series_metadata(path):
    """Read the first TIFF series' axes and shape with actionable validation."""
    try:
        with tifffile.TiffFile(path) as tif:
            if not tif.series:
                raise ValueError("contains no image series")
            series = tif.series[0]
            axes, shape = series.axes, series.shape
    except (OSError, tifffile.TiffFileError) as exc:
        raise ValueError(f"Could not read TIFF `{path}`: {exc}") from exc
    if len(axes) != len(shape) or axes.count("Y") != 1 or axes.count("X") != 1:
        raise ValueError(f"TIFF `{path}` must declare exactly one Y and X axis; found axes `{axes}`.")
    unsupported = [(axis, length) for axis, length in zip(axes, shape) if axis not in SUPPORTED_AXES and length != 1]
    if unsupported:
        raise ValueError(f"TIFF `{path}` has unsupported non-singleton axes {unsupported}; expected only T, C, Z, Y, X.")
    return axes, shape


def available_channel_indices(source_directory):
    """Return channel indices available in every TIFF in a source directory."""
    counts = []
    for path in find_tiff_files(source_directory):
        axes, shape = _read_series_metadata(path)
        counts.append(shape[axes.index("C")] if "C" in axes else 1)
    return list(range(min(counts)))


def _timepoints_from_tiff(path, channel_index):
    """Yield 2-D intensity frames after selecting C and maximum-projecting Z."""
    axes, _ = _read_series_metadata(path)
    try:
        with tifffile.TiffFile(path) as tif:
            image = tif.series[0].asarray()
    except (OSError, tifffile.TiffFileError) as exc:
        raise ValueError(f"Could not read TIFF pixels from `{path}`: {exc}") from exc

    image = np.asarray(image)
    for axis in tuple(axes):
        if axis not in SUPPORTED_AXES:
            index = axes.index(axis)
            image = np.take(image, 0, axis=index)
            axes = axes[:index] + axes[index + 1:]
    if "C" in axes:
        index = axes.index("C")
        if not 0 <= channel_index < image.shape[index]:
            raise ValueError(f"Channel {channel_index} is unavailable in TIFF `{path}` (has {image.shape[index]} channels).")
        image = np.take(image, channel_index, axis=index)
        axes = axes[:index] + axes[index + 1:]
    elif channel_index != 0:
        raise ValueError(f"TIFF `{path}` has no channel axis; only channel 0 can be selected.")
    if "Z" in axes:
        index = axes.index("Z")
        image = np.max(image, axis=index)
        axes = axes[:index] + axes[index + 1:]
    if axes == "YX":
        image, axes = image[np.newaxis, ...], "TYX"
    if axes != "TYX":
        raise ValueError(f"TIFF `{path}` could not be reduced to TYX; remaining axes are `{axes}`.")
    yield from image


def _series_directory_name(path, used_names):
    """Create a deterministic safe, unique directory name from a TIFF filename."""
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._") or "series"
    name, suffix = base, 2
    while name.casefold() in used_names:
        name, suffix = f"{base}_{suffix}", suffix + 1
    used_names.add(name.casefold())
    return name


def _write_project(tiff_paths, project_directory, channel_index, progress_callback=None):
    """Write a complete project into an already-created temporary directory."""
    images_directory = project_directory / "images"
    images_directory.mkdir()
    ElementTree.ElementTree(ElementTree.Element("annotations")).write(images_directory / CELL_REGIONS_FILENAME, encoding="utf-8", xml_declaration=True)
    used_names, frame_count = set(), 0
    total_frames = sum(1 for path in tiff_paths for _ in _timepoints_from_tiff(path, channel_index))
    completed = 0
    for path in tiff_paths:
        series_directory = images_directory / _series_directory_name(path, used_names)
        series_directory.mkdir()
        for timepoint, frame in enumerate(_timepoints_from_tiff(path, channel_index), start=1):
            Image.fromarray(preprocess_image(frame)).save(series_directory / f"t{timepoint:03}.png")
            completed += 1
            frame_count += 1
            if progress_callback:
                progress_callback(completed, total_frames, str(path))
    return {"series": len(tiff_paths), "frames": frame_count, "project_directory": str(project_directory)}


def create_projects_from_tiff_folder(source_directory, output_directory, channel_index=0, separate_projects=False, progress_callback=None):
    """Create atomic CellClicker project(s) from every TIFF in one directory."""
    tiff_paths = find_tiff_files(source_directory)
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_directory}")
    if channel_index not in available_channel_indices(source_directory):
        raise ValueError(f"Channel {channel_index} is not available in every TIFF in {source_directory}.")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="celfdrive-tiff-import-", dir=output_directory.parent))
    try:
        if separate_projects:
            summaries, used_names = [], set()
            for path in tiff_paths:
                name = _series_directory_name(path, used_names)
                project = staging / name
                project.mkdir()
                summaries.append(_write_project([path], project, channel_index, progress_callback))
            result = {"projects": summaries, "series": len(tiff_paths), "frames": sum(item["frames"] for item in summaries)}
        else:
            result = _write_project(tiff_paths, staging, channel_index, progress_callback)
            result["projects"] = [result.copy()]
        os.replace(staging, output_directory)
        for project in result["projects"]:
            project["project_directory"] = project["project_directory"].replace(str(staging), str(output_directory), 1)
        result["output_directory"] = str(output_directory)
        if not separate_projects:
            result["project_directory"] = str(output_directory)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
