from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import tifffile

from CellClicker.image_series import discover_image_series
from CellClicker.tiff_project_import import create_projects_from_tiff_folder


def _write_tiff(path, data, axes):
    tifffile.imwrite(path, data, metadata={"axes": axes})


def test_imports_tyx_tiff_as_normalized_timepoints(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_tiff(source / "position 1.tif", np.array([[[0, 1], [2, 3]], [[10, 20], [30, 40]]], dtype=np.uint16), "TYX")

    result = create_projects_from_tiff_folder(source, tmp_path / "project")

    images = tmp_path / "project" / "images" / "position_1"
    assert result["series"] == 1
    assert result["frames"] == 2
    assert (tmp_path / "project" / "images" / "cell_regions.xml").is_file()
    assert [path.name for path in images.iterdir()] == ["t001.png", "t002.png"]
    assert np.array_equal(np.asarray(Image.open(images / "t001.png")), [[0, 85], [170, 255]])


def test_import_selects_channel_and_max_projects_z(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    data = np.zeros((2, 2, 3, 2, 2), dtype=np.uint16)
    data[:, 1, :, :, :] = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[2, 3], [4, 5]]], [[[10, 20], [30, 40]], [[50, 60], [70, 80]], [[20, 30], [40, 50]]]])
    _write_tiff(source / "P1.tiff", data, "TCZYX")

    create_projects_from_tiff_folder(source, tmp_path / "project", channel_index=1)

    image = np.asarray(Image.open(tmp_path / "project" / "images" / "P1" / "t002.png"))
    assert np.array_equal(image, [[0, 85], [170, 255]])


def test_import_max_projects_tzyx_tiff(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    data = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.uint16)
    _write_tiff(source / "P1.tif", data, "TZYX")

    create_projects_from_tiff_folder(source, tmp_path / "project")

    image = np.asarray(Image.open(tmp_path / "project" / "images" / "P1" / "t001.png"))
    assert np.array_equal(image, [[0, 85], [170, 255]])


def test_import_separate_projects_and_refuses_existing_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_tiff(source / "a.tif", np.ones((2, 2), dtype=np.uint16), "YX")
    _write_tiff(source / "b.tif", np.ones((2, 2), dtype=np.uint16), "YX")

    output = tmp_path / "projects"
    result = create_projects_from_tiff_folder(source, output, separate_projects=True)

    assert result["series"] == 2
    assert (output / "a" / "images" / "a" / "t001.png").is_file()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_projects_from_tiff_folder(source, output)


def test_import_rejects_invalid_channel_without_creating_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_tiff(source / "a.tif", np.ones((1, 2, 2, 2), dtype=np.uint16), "TZYX")

    with pytest.raises(ValueError, match="not available"):
        create_projects_from_tiff_folder(source, tmp_path / "project", channel_index=1)
    assert not (tmp_path / "project").exists()


def test_discover_image_series_orders_timepoints_without_crossing_series(tmp_path):
    images = tmp_path / "images"
    for name in ("position_b/t010.png", "position_b/t002.png", "position_a/t001.png"):
        path = images / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")

    discovered = discover_image_series(images)

    assert list(discovered) == ["position_a", "position_b"]
    assert [Path(path).name for path in discovered["position_b"]] == ["t002.png", "t010.png"]
