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


def _timepoint_count(path):
    """Return the number of timepoints a TIFF contributes without reading pixels."""
    axes, shape = _read_series_metadata(path)
    return shape[axes.index("T")] if "T" in axes else 1


def series_name_for_tiff(path, used_names):
    """Create a deterministic, filesystem-safe, unique series name for a TIFF.

    The name is used as the filename prefix of every frame the TIFF produces,
    so it preserves the source stem and is disambiguated when two stems
    sanitize identically.
    """
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(path).stem).strip("._") or "series"
    name, suffix = base, 2
    while name.casefold() in used_names:
        name, suffix = f"{base}_{suffix}", suffix + 1
    used_names.add(name.casefold())
    return name


def assign_series_names(tiff_paths):
    """Return ``{tiff path: series name}`` for one import, resolving collisions."""
    used_names = set()
    return {path: series_name_for_tiff(path, used_names) for path in tiff_paths}


def frame_filename(series_name, timepoint, frame_count=0):
    """Return the flat project filename for one timepoint of one series.

    Frame numbers are zero padded to at least three digits so that they sort
    chronologically as text and match CellClicker's ``t<frame>`` convention.
    """
    return f"{series_name}_t{timepoint:0{max(3, len(str(frame_count)))}d}.png"


def _write_project(tiff_paths, project_directory, channel_index, series_names, progress_callback=None):
    """Write a complete flat project into an already-created temporary directory."""
    images_directory = project_directory / "images"
    images_directory.mkdir()
    ElementTree.ElementTree(ElementTree.Element("annotations")).write(images_directory / CELL_REGIONS_FILENAME, encoding="utf-8", xml_declaration=True)
    frame_counts = {path: _timepoint_count(path) for path in tiff_paths}
    total_frames = sum(frame_counts.values())
    completed = 0
    for path in tiff_paths:
        series_name, frame_count = series_names[path], frame_counts[path]
        for timepoint, frame in enumerate(_timepoints_from_tiff(path, channel_index), start=1):
            Image.fromarray(preprocess_image(frame)).save(images_directory / frame_filename(series_name, timepoint, frame_count))
            completed += 1
            if progress_callback:
                progress_callback(completed, total_frames, str(path))
    return {"series": len(tiff_paths), "frames": completed, "project_directory": str(project_directory)}


def create_projects_from_tiff_folder(source_directory, output_directory, channel_index=0, separate_projects=False, progress_callback=None):
    """Create atomic CellClicker project(s) from every TIFF in one directory.

    Every generated PNG is written directly into the project's flat ``images/``
    directory as ``<series>_t<frame>.png``; series identity lives in the
    filename rather than in a directory hierarchy.
    """
    tiff_paths = find_tiff_files(source_directory)
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_directory}")
    if channel_index not in available_channel_indices(source_directory):
        raise ValueError(f"Channel {channel_index} is not available in every TIFF in {source_directory}.")
    series_names = assign_series_names(tiff_paths)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="celfdrive-tiff-import-", dir=output_directory.parent))
    try:
        if separate_projects:
            summaries = []
            for path in tiff_paths:
                project = staging / series_names[path]
                project.mkdir()
                summaries.append(_write_project([path], project, channel_index, series_names, progress_callback))
            result = {"projects": summaries, "series": len(tiff_paths), "frames": sum(item["frames"] for item in summaries)}
        else:
            result = _write_project(tiff_paths, staging, channel_index, series_names, progress_callback)
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
