"""Create CellClicker project(s) from a folder of metadata-labelled TIFF time series."""

import argparse

from CellClicker.tiff_project_import import create_projects_from_tiff_folder
from benchmarks.core import preprocess_image


def create_project(source_position_directory, output_project, channel_index=0):
    """Backward-compatible wrapper creating one project from TIFFs in a folder."""
    return create_projects_from_tiff_folder(source_position_directory, output_project, channel_index=channel_index)["frames"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tiff-directory", "--source-position-directory", dest="source_directory", required=True)
    parser.add_argument("--output-project", required=True)
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument("--separate-projects", action="store_true")
    args = parser.parse_args()
    result = create_projects_from_tiff_folder(
        args.source_directory, args.output_project, args.channel_index, args.separate_projects,
    )
    print(f"Created {result['series']} series / {result['frames']} frames in {result['output_directory']}")
