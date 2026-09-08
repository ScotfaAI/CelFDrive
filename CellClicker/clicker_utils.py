"""Path, image-series, and YOLO bounding-box utilities for CellClicker.

Pixel boxes use ``(x_min, y_min, x_max, y_max)`` or ``(x, y, width, height)``
as named by each function; YOLO values are normalized to image dimensions.
"""

import re
import os
from functools import lru_cache

def convert_path_format(input_path):
    """Return ``input_path`` with separators normalized for the current OS."""
    if os.name == 'posix':
        # Running on macOS or Linux, convert to macOS-style path
        return input_path.replace('\\', '/')
    else:
        # Running on Windows, convert to Windows-style path
        return input_path.replace('/', '\\')

#: Trailing timepoint of an image or label filename. Reading is permissive:
#: the marker may be ``t`` or ``T`` with any number of digits. It may not
#: follow a letter, so words ending in ``t`` and digits (``slot12``) are not
#: mistaken for timepoints. Rewrites reproduce the marker and width they read,
#: so a stepped-back name still matches the file on disk.
def _timepoint_pattern(extension):
    return re.compile(rf'(?<![A-Za-z])([tT])(\d+)\.{extension}$')


IMAGE_TIMEPOINT_PATTERN = _timepoint_pattern("png")
LABEL_TIMEPOINT_PATTERN = _timepoint_pattern("txt")


@lru_cache(maxsize=64)
def _series_minimum_width(directory, prefix, marker, extension):
    """Return the narrowest timepoint width used by one series on disk.

    A name such as ``expt_P01_t100.png`` cannot say on its own whether its
    series is unpadded (so the frame before it is ``t99``) or padded to three
    digits (``t099``). The sibling files in the same folder settle it, and this
    is only consulted when a step crosses down a power of ten.
    """
    sibling = re.compile(re.escape(prefix) + re.escape(marker) + rf'(\d+)\.{extension}$')
    try:
        entries = os.listdir(directory or '.')
    except OSError:
        return None
    widths = [len(match.group(1)) for match in map(sibling.match, entries) if match]
    return min(widths) if widths else None


def clear_series_width_cache():
    """Forget cached series padding, after images may have been renamed."""
    _series_minimum_width.cache_clear()


def _step_timepoint(name, pattern, extension, new_timepoint):
    """Rewrite a filename's timepoint, keeping its letter case and padding.

    A series is padded to at least the narrowest width it uses, so the rewritten
    name matches a file that exists whether the series counts ``t1, t2, ...`` or
    ``t001, t002, ...``.
    """
    match = pattern.search(name)
    marker, digits = match.group(1), match.group(2)
    width, natural = len(digits), len(str(new_timepoint))
    if natural < width and not digits.startswith('0'):
        # Stepping below a power of ten: the name alone cannot say whether the
        # series is padded to this width or simply reached this many digits.
        directory, filename = os.path.split(name)
        prefix = filename[: pattern.search(filename).start(1)]
        minimum = _series_minimum_width(directory, prefix, marker, extension)
        if minimum is not None:
            width = max(minimum, natural)
    return pattern.sub(f'{marker}{new_timepoint:0{width}d}.{extension}', name)


def get_previous_image_name(image_name):
    """Decrease the timepoint of the image by one and return the new image name.

    Returns ``None`` at the first frame of a series, and for names that carry
    no timepoint at all.
    """
    match = IMAGE_TIMEPOINT_PATTERN.search(image_name)
    if not match:
        return None  # Return None if the format doesn't match
    timepoint = int(match.group(2)) - 1
    # If timepoint would become 0, return None
    if timepoint == 0:
        return None
    return _step_timepoint(image_name, IMAGE_TIMEPOINT_PATTERN, "png", timepoint)


def get_relative_image_name(image_name, stepback):
    """Return the image filename ``stepback`` frames before a series filename.

    The series prefix, timepoint letter case and digit width of ``image_name``
    are preserved, so backtracking never leaves the series the frame belongs to
    and never invents a name that is not on disk.
    """
    match = IMAGE_TIMEPOINT_PATTERN.search(image_name)
    if not match:
        return None  # Return None if the format doesn't match
    timepoint = int(match.group(2)) - stepback
    # If adjusted timepoint is less than 1 return None
    if timepoint < 1:
        return None
    return _step_timepoint(image_name, IMAGE_TIMEPOINT_PATTERN, "png", timepoint)


def get_relative_label_name(image_name, stepback):
    """Decrease the timepoint of the label by stepback and return the new name.

    The series prefix, timepoint letter case and digit width are preserved.
    """
    match = LABEL_TIMEPOINT_PATTERN.search(image_name)
    if not match:
        return None  # Return None if the format doesn't match
    # If timepoint is already 0, keep it as is (or handle accordingly)
    timepoint = max(0, int(match.group(2)) - stepback)
    return _step_timepoint(image_name, LABEL_TIMEPOINT_PATTERN, "txt", timepoint)


def append_yolov5_label(label_path, x_center, y_center, width, height, img_width, img_height, class_id):
    """Append one pixel-space box as a normalized YOLO label line.

    ``x_center`` and ``y_center`` are image pixels; width/height are pixel
    extents; ``img_width``/``img_height`` define the normalization dimensions.
    """
    """Append a new YOLOv5 label to a label file, removing any existing newline characters."""
    # Normalize the coordinates
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    # Read the existing content of the file
    with open(label_path, 'r') as f:
        existing_content = f.read()

    # Remove newline characters if they exist
    existing_content = existing_content.replace("    \n", "")

    # Append the new label and add a newline
    new_label = f"{class_id} {x_center} {y_center} {width} {height}\n"

    # Append the new label to the file
    with open(label_path, 'a') as f:
        f.write(new_label)
        
def yolov5_to_xyxy(x_center, y_center, width, height, image_width, image_height):
    """Convert a normalized YOLO centre box to pixel ``(x_min, y_min, x_max, y_max)``."""
    # Convert YOLOv5 coordinates to real coordinates
    x_center_real = x_center * image_width
    y_center_real = y_center * image_height
    box_width = width * image_width
    box_height = height * image_height
    
    # Calculate top-left corner coordinates
    x1 = x_center_real - (box_width / 2)
    y1 = y_center_real - (box_height / 2)
    
    # Calculate bottom-right corner coordinates
    x2 = x_center_real + (box_width / 2)
    y2 = y_center_real + (box_height / 2)
    
    return [x1, y1, x2, y2]

def yolov5_to_xywh(x_center, y_center, width, height, image_width, image_height):
    """Convert a normalized YOLO centre box to pixel ``(x, y, width, height)``."""
    # Convert YOLOv5 coordinates to real coordinates
    x_center_real = x_center * image_width
    y_center_real = y_center * image_height
    box_width = width * image_width
    box_height = height * image_height
    
    # Calculate top-left corner coordinates
    x1 = x_center_real - (box_width / 2)
    y1 = y_center_real - (box_height / 2)
    
    # Return the bounding box in xywh format
    return [x1, y1, box_width, box_height]
