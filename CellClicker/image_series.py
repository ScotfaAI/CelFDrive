"""Discover and order the logical image time series inside a CellClicker project.

CellClicker projects keep every image directly inside ``images/``. A frame's
logical series is therefore carried by its filename, using the project
convention ``<experiment>_P<position>_t<frame>.png``, and not by any directory
hierarchy.

Reading is deliberately permissive: the timepoint may be written ``t`` or
``T``, padded to any width or not padded at all, and the position may be
spelled ``P01``, ``p1`` or omitted entirely.  A series is read as "zero padded
to at least the narrowest width it uses", so ``t1 ... t200`` and
``t001 ... t200`` are both understood.  Newly generated projects are always
written in the canonical zero-padded lower-case form.
"""

from collections import OrderedDict
from pathlib import Path
import re


IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".gif"})
DEFAULT_SERIES_NAME = "default"

#: ``<series>_t<frame>`` filename stems, e.g. ``P0037_t001``, ``expt_P1_T27``
#: or ``P1 - 1t027``.  The timepoint marker may not follow a letter, so words
#: that merely end in ``t`` and digits (``slot12``, ``point7``) are not read as
#: timepoints.
FRAME_PATTERN = re.compile(r"^(?P<series>.*?)_?(?<![A-Za-z])(?P<marker>[tT])(?P<frame>\d+)$")


class UnsupportedFrameNamingError(ValueError):
    """Raised when a series' filenames cannot be stepped through backwards."""


def split_series_and_frame(filename):
    """Return ``(series-name, frame-number)`` parsed from one image filename.

    Returns ``(None, None)`` for names that carry no timepoint, which lets
    projects with no series convention fall back to a single series.
    """
    match = FRAME_PATTERN.match(Path(filename).stem)
    if not match:
        return None, None
    return (match.group("series") or DEFAULT_SERIES_NAME), int(match.group("frame"))


def frame_marker_and_digits(filename):
    """Return the ``(marker, digit text)`` a filename writes its timepoint in."""
    match = FRAME_PATTERN.match(Path(filename).stem)
    if not match:
        return None
    return match.group("marker"), match.group("frame")


def series_minimum_width(filenames):
    """Return the narrowest timepoint width used across a series' filenames.

    A series is written by zero padding each frame number to at least this
    width, so ``t1 ... t200`` has a minimum width of 1 and ``t001 ... t200`` a
    minimum width of 3.  Returns ``None`` when no filename carries a timepoint.
    """
    widths = [len(found[1]) for found in map(frame_marker_and_digits, filenames) if found]
    return min(widths) if widths else None


def check_frame_styles(series_name, filenames):
    """Raise a readable error when a series' timepoints are not one pattern.

    Frames are read as ``zero padded to at least the narrowest width used``,
    which accepts unpadded (``t1 ... t200``), narrowly padded (``t01 ... t100``)
    and conventionally padded (``t001 ... t200``) series alike.  What cannot be
    read is a series that mixes patterns, because the only way to reach an
    earlier frame is to rewrite the timepoint in the filename.
    """
    timed = [(Path(name).name, *frame_marker_and_digits(name)) for name in filenames if frame_marker_and_digits(name)]
    if not timed:
        return

    markers = sorted({marker for _, marker, _ in timed})
    if len(markers) > 1:
        examples = "\n".join(
            f"    `{marker}`: {next(name for name, found, _ in timed if found == marker)}"
            for marker in markers
        )
        raise UnsupportedFrameNamingError(
            f"The images for series `{series_name}` mark the timepoint with both "
            f"`{markers[0]}` and `{markers[1]}`:\n{examples}\n\n"
            "CellClicker reaches an earlier frame by rewriting the timepoint in the "
            "filename, so every image in one series must use the same letter. Rename "
            "them to one form, then load the project again."
        )

    minimum = min(len(digits) for _, _, digits in timed)
    offenders = [name for name, _, digits in timed if len(digits) != max(minimum, len(str(int(digits))))]
    if not offenders:
        return
    listed = "\n".join(f"    {name}" for name in sorted(offenders)[:3])
    remaining = f"\n    ...and {len(offenders) - 3} more" if len(offenders) > 3 else ""
    example = f"{markers[0]}{'0' * (minimum - 1)}1"
    raise UnsupportedFrameNamingError(
        f"The images for series `{series_name}` do not zero pad the timepoint "
        f"consistently. Most are padded to at least {minimum} digit(s), like "
        f"`{example}`, but these are not:\n{listed}{remaining}\n\n"
        "CellClicker reaches an earlier frame by rewriting the timepoint in the "
        "filename, so a series must pad every frame to the same minimum width. "
        "Rename the images above to match, then load the project again."
    )


def series_name_for_image(images_directory, path):
    """Return the logical series a discovered image belongs to.

    Flat images are grouped by their filename prefix. Images inside a
    subdirectory are grouped by that subdirectory, which keeps projects
    generated by the superseded nested importer readable; new projects are
    always written flat.
    """
    relative = Path(path).relative_to(images_directory)
    if relative.parent != Path("."):
        return relative.parent.as_posix()
    series, _ = split_series_and_frame(relative.name)
    return series or DEFAULT_SERIES_NAME


def natural_sort_key(name):
    """Order names so that digit runs compare numerically and case is ignored.

    Position tokens are written inconsistently across acquisitions (``P01``,
    ``p1``, ``P1``), so ``p2`` must order before ``p10`` rather than after it.
    The whole name is appended as a tie-breaker, keeping names that differ only
    in zero padding or case in a stable, deterministic order.
    """
    parts = [
        (int(part), "") if part.isdigit() else (float("inf"), part.casefold())
        for part in re.split(r"(\d+)", name)
    ]
    return parts + [(float("inf"), name.casefold())]


def series_menu_labels(series_names):
    """Return ``{series name: short display label}`` for a series selector.

    Project filenames repeat the whole experiment name in every series, so the
    leading underscore-separated tokens that every series shares are dropped to
    leave the part that actually distinguishes them, usually the position.
    Full names are kept whenever shortening would be ambiguous.
    """
    series_names = list(series_names)
    if len(series_names) < 2:
        return {name: name for name in series_names}
    token_lists = [name.split("_") for name in series_names]
    shared = 0
    while all(len(tokens) > shared + 1 for tokens in token_lists) and len({tokens[shared] for tokens in token_lists}) == 1:
        shared += 1
    labels = {name: "_".join(tokens[shared:]) for name, tokens in zip(series_names, token_lists)}
    if len(set(labels.values())) != len(series_names) or not all(labels.values()):
        return {name: name for name in series_names}
    return labels


def discover_image_series(images_directory):
    """Return ``series-name -> ordered absolute image paths`` for a project.

    Series are ordered naturally by name and frames within a series by
    timepoint, with a filename tie-breaker for images that carry no timepoint.

    Raises :class:`UnsupportedFrameNamingError` when one series mixes timepoint
    patterns, rather than loading a project whose backward tracking would
    silently produce filenames that do not exist.
    """
    images_directory = Path(images_directory)
    grouped = {}
    for path in images_directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        grouped.setdefault(series_name_for_image(images_directory, path), []).append(path)

    def frame_key(path):
        _, frame = split_series_and_frame(path.name)
        return (float("inf") if frame is None else frame, path.name.casefold())

    for series, paths in grouped.items():
        check_frame_styles(series, [path.name for path in paths])

    return OrderedDict(
        (series, [str(path) for path in sorted(paths, key=frame_key)])
        for series, paths in sorted(grouped.items(), key=lambda item: natural_sort_key(item[0]))
    )
