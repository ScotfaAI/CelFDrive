"""Discover and order CellClicker image time series stored below ``images/``."""

from collections import OrderedDict
from pathlib import Path
import re


IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".gif"})
TIMEPOINT_PATTERN = re.compile(r"t(\d+)\.[^.]+$", re.IGNORECASE)


def discover_image_series(images_directory):
    """Return ``series-name -> ordered absolute image paths`` recursively.

    A direct image beneath ``images/`` belongs to ``default``; images beneath a
    subdirectory belong to that relative subdirectory.  Timepoint names are
    sorted numerically before a filename tie-breaker.
    """
    images_directory = Path(images_directory)
    grouped = {}
    for path in images_directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(images_directory)
        series = relative.parent.as_posix() if relative.parent != Path(".") else "default"
        grouped.setdefault(series, []).append(path)

    def frame_key(path):
        match = TIMEPOINT_PATTERN.search(path.name)
        return (int(match.group(1)) if match else float("inf"), path.name.casefold())

    return OrderedDict((series, [str(path) for path in sorted(paths, key=frame_key)]) for series, paths in sorted(grouped.items(), key=lambda item: item[0].casefold()))
