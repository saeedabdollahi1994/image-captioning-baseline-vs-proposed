"""Tests for Flickr8k image-caption datasets.

No real Flickr8k files are required. Every test creates a small temporary
caption file, split file, and image directory.

Run from the project root with:

    python -m pytest tests/test_dataset.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest
import torch
from PIL import Image
from torch import Tensor

from src.data.dataset import (
    ImageCaptionEvaluationDataset,
    ImageCaptionTrainingDataset,
    _load_image,
    _parse_caption_row,
    load_split_image_names,
    read_captions_file,
    read_shared_split,
)


IMAGE_SIZE = 8

TRAIN_IMAGE_1 = "train_1.jpg"
TRAIN_IMAGE_2 = "train_2.jpg"
VAL_IMAGE = "val_1.jpg"
TEST_IMAGE = "test_1.jpg"


def tensor_transform(image: Image.Image) -> Tensor:
    """Return a deterministic 3-channel Tensor for a PIL image."""
    assert image.mode == "RGB"
    return torch.ones(
        3,
        image.height,
        image.width,
        dtype=torch.float32,
    )


def write_image(
    path: Path,
    *,
    mode: str = "RGB",
) -> None:
    """Create a small valid image file."""
    if mode == "RGB":
        color: int | tuple[int, int, int] = (10, 20, 30)
    else:
        color = 127

    image = Image.new(
        mode,
        (IMAGE_SIZE, IMAGE_SIZE),
        color=color,
    )
    image.save(path)


def write_captions_file(path: Path) -> None:
    """Create a small Flickr8k-style CSV caption file."""
    path.write_text(
        "image,caption\n"
        f"{TRAIN_IMAGE_1},first train caption\n"
        f"{TRAIN_IMAGE_1},second train caption, with comma\n"
        f"{TRAIN_IMAGE_2},caption for second train image\n"
        f"{VAL_IMAGE},validation caption\n"
        f"{TEST_IMAGE},test caption one\n"
        f"{TEST_IMAGE},test caption two\n",
        encoding="utf-8",
    )


def write_split_file(
    path: Path,
    *,
    train_images: list[object] | None = None,
    val_images: list[object] | None = None,
    test_images: list[object] | None = None,
) -> None:
    """Write a shared split JSON file."""
    payload = {
        "train_images": (
            [TRAIN_IMAGE_1, TRAIN_IMAGE_2]
            if train_images is None
            else train_images
        ),
        "val_images": (
            [VAL_IMAGE]
            if val_images is None
            else val_images
        ),
        "test_images": (
            [TEST_IMAGE]
            if test_images is None
            else test_images
        ),
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


@pytest.fixture
def dataset_files(
    tmp_path: Path,
) -> dict[str, Path]:
    """Create a complete temporary dataset layout."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    for image_name in (
        TRAIN_IMAGE_1,
        TRAIN_IMAGE_2,
        VAL_IMAGE,
        TEST_IMAGE,
    ):
        write_image(images_dir / image_name)

    captions_path = tmp_path / "captions.txt"
    split_path = tmp_path / "shared_split.json"

    write_captions_file(captions_path)
    write_split_file(split_path)

    return {
        "images_dir": images_dir,
        "captions_path": captions_path,
        "split_path": split_path,
    }


def test_parse_caption_row_accepts_csv_format() -> None:
    """CSV rows must preserve commas inside the caption."""
    image_name, caption = _parse_caption_row(
        "image.jpg,a caption, with a comma",
        line_number=3,
    )

    assert image_name == "image.jpg"
    assert caption == "a caption, with a comma"


def test_parse_caption_row_accepts_legacy_tab_format() -> None:
    """Legacy Flickr8k image.jpg#N rows must be supported."""
    image_name, caption = _parse_caption_row(
        "image.jpg#4\ta legacy caption",
        line_number=7,
    )

    assert image_name == "image.jpg"
    assert caption == "a legacy caption"


def test_parse_caption_row_rejects_unknown_format() -> None:
    """Rows without a comma or tab must be rejected."""
    with pytest.raises(ValueError, match="Invalid caption row"):
        _parse_caption_row(
            "invalid caption row",
            line_number=2,
        )


@pytest.mark.parametrize(
    "row",
    [
        ",caption without image",
        "   ,caption without image",
    ],
)
def test_parse_caption_row_rejects_missing_image_name(
    row: str,
) -> None:
    """Every caption row must contain an image name."""
    with pytest.raises(ValueError, match="Missing image name"):
        _parse_caption_row(row, line_number=4)


@pytest.mark.parametrize(
    "row",
    [
        "image.jpg,",
        "image.jpg,   ",
    ],
)
def test_parse_caption_row_rejects_missing_caption(
    row: str,
) -> None:
    """Every image row must contain a non-empty caption."""
    with pytest.raises(ValueError, match="Missing caption"):
        _parse_caption_row(row, line_number=5)


def test_read_captions_file_groups_captions_and_skips_header(
    dataset_files: dict[str, Path],
) -> None:
    """Captions must be grouped by filename in original order."""
    captions = read_captions_file(
        dataset_files["captions_path"]
    )

    assert set(captions) == {
        TRAIN_IMAGE_1,
        TRAIN_IMAGE_2,
        VAL_IMAGE,
        TEST_IMAGE,
    }

    assert captions[TRAIN_IMAGE_1] == [
        "first train caption",
        "second train caption, with comma",
    ]

    assert captions[TEST_IMAGE] == [
        "test caption one",
        "test caption two",
    ]


def test_read_captions_file_skips_blank_lines(
    tmp_path: Path,
) -> None:
    """Empty rows must not create invalid caption records."""
    captions_path = tmp_path / "captions.txt"
    captions_path.write_text(
        "image,caption\n\n"
        "image.jpg,caption one\n"
        "   \n"
        "image.jpg,caption two\n",
        encoding="utf-8",
    )

    captions = read_captions_file(captions_path)

    assert captions == {
        "image.jpg": ["caption one", "caption two"]
    }


def test_read_captions_file_supports_utf8_bom(
    tmp_path: Path,
) -> None:
    """UTF-8 BOM in exported caption files must be handled."""
    captions_path = tmp_path / "captions.txt"
    captions_path.write_text(
        "\ufeffimage,caption\nimage.jpg,a caption\n",
        encoding="utf-8",
    )

    captions = read_captions_file(captions_path)

    assert captions == {"image.jpg": ["a caption"]}


def test_read_captions_file_rejects_empty_file(
    tmp_path: Path,
) -> None:
    """A caption file without records must fail early."""
    captions_path = tmp_path / "captions.txt"
    captions_path.write_text(
        "image,caption\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No valid captions"):
        read_captions_file(captions_path)


def test_read_captions_file_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """A missing caption file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_captions_file(tmp_path / "missing.txt")


def test_read_captions_file_rejects_directory_path(
    tmp_path: Path,
) -> None:
    """captions_path must refer to a regular file."""
    with pytest.raises(ValueError, match="regular file"):
        read_captions_file(tmp_path)


def test_read_shared_split_returns_normalized_splits(
    dataset_files: dict[str, Path],
) -> None:
    """Split keys must be converted to train, val, and test."""
    splits = read_shared_split(
        dataset_files["split_path"]
    )

    assert splits == {
        "train": [TRAIN_IMAGE_1, TRAIN_IMAGE_2],
        "val": [VAL_IMAGE],
        "test": [TEST_IMAGE],
    }


def test_read_shared_split_rejects_overlap(
    tmp_path: Path,
) -> None:
    """The same image must never appear in multiple splits."""
    split_path = tmp_path / "shared_split.json"
    write_split_file(
        split_path,
        train_images=[TRAIN_IMAGE_1],
        val_images=[TRAIN_IMAGE_1],
        test_images=[TEST_IMAGE],
    )

    with pytest.raises(ValueError, match="Data leakage detected"):
        read_shared_split(split_path)


def test_read_shared_split_rejects_duplicates_inside_split(
    tmp_path: Path,
) -> None:
    """A split must not contain duplicate image names."""
    split_path = tmp_path / "shared_split.json"
    write_split_file(
        split_path,
        train_images=[TRAIN_IMAGE_1, TRAIN_IMAGE_1],
    )

    with pytest.raises(ValueError, match="Duplicate image names"):
        read_shared_split(split_path)


def test_read_shared_split_rejects_missing_key(
    tmp_path: Path,
) -> None:
    """All three split keys are required."""
    split_path = tmp_path / "shared_split.json"
    split_path.write_text(
        json.dumps(
            {
                "train_images": [TRAIN_IMAGE_1],
                "val_images": [VAL_IMAGE],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="test_images"):
        read_shared_split(split_path)


def test_read_shared_split_rejects_non_object_root(
    tmp_path: Path,
) -> None:
    """The JSON root must be an object rather than a list."""
    split_path = tmp_path / "shared_split.json"
    split_path.write_text(
        json.dumps([TRAIN_IMAGE_1]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="root must be an object"):
        read_shared_split(split_path)


def test_read_shared_split_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """Malformed JSON must produce a clear ValueError."""
    split_path = tmp_path / "shared_split.json"
    split_path.write_text(
        "{not valid json",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid JSON"):
        read_shared_split(split_path)


def test_read_shared_split_rejects_non_list_value(
    tmp_path: Path,
) -> None:
    """Each split value must be a list of image names."""
    split_path = tmp_path / "shared_split.json"
    payload = {
        "train_images": TRAIN_IMAGE_1,
        "val_images": [VAL_IMAGE],
        "test_images": [TEST_IMAGE],
    }
    split_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="must contain a list"):
        read_shared_split(split_path)


def test_read_shared_split_rejects_non_string_item(
    tmp_path: Path,
) -> None:
    """Every split item must be a string filename."""
    split_path = tmp_path / "shared_split.json"
    write_split_file(
        split_path,
        train_images=[TRAIN_IMAGE_1, 123],
    )

    with pytest.raises(TypeError, match="must be a string"):
        read_shared_split(split_path)


def test_read_shared_split_rejects_blank_image_name(
    tmp_path: Path,
) -> None:
    """Empty image names are invalid split members."""
    split_path = tmp_path / "shared_split.json"
    write_split_file(
        split_path,
        train_images=["   "],
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        read_shared_split(split_path)


def test_load_split_image_names_normalizes_name_and_returns_copy(
    dataset_files: dict[str, Path],
) -> None:
    """Whitespace and letter case in split_name must be normalized."""
    image_names = load_split_image_names(
        dataset_files["split_path"],
        "  TRAIN  ",
    )

    assert image_names == [TRAIN_IMAGE_1, TRAIN_IMAGE_2]

    image_names.append("new.jpg")

    second_read = load_split_image_names(
        dataset_files["split_path"],
        "train",
    )
    assert "new.jpg" not in second_read


def test_load_split_image_names_rejects_non_string_name(
    dataset_files: dict[str, Path],
) -> None:
    """split_name must be a string."""
    with pytest.raises(TypeError, match="split_name must be a string"):
        load_split_image_names(
            dataset_files["split_path"],
            1,  # type: ignore[arg-type]
        )


def test_load_split_image_names_rejects_unknown_name(
    dataset_files: dict[str, Path],
) -> None:
    """Only train, val, and test are accepted."""
    with pytest.raises(ValueError, match="must be 'train', 'val', or 'test'"):
        load_split_image_names(
            dataset_files["split_path"],
            "validation",
        )


def test_training_dataset_creates_one_sample_per_caption(
    dataset_files: dict[str, Path],
) -> None:
    """Training length must equal captions belonging to train images."""
    dataset = ImageCaptionTrainingDataset(
        images_dir=dataset_files["images_dir"],
        captions_path=dataset_files["captions_path"],
        split_path=dataset_files["split_path"],
        split_name="train",
        transform=tensor_transform,
    )

    assert len(dataset) == 3
    assert dataset.samples == [
        (TRAIN_IMAGE_1, "first train caption"),
        (TRAIN_IMAGE_1, "second train caption, with comma"),
        (TRAIN_IMAGE_2, "caption for second train image"),
    ]


def test_training_dataset_getitem_returns_expected_values(
    dataset_files: dict[str, Path],
) -> None:
    """Training samples must contain image Tensor, caption, and filename."""
    dataset = ImageCaptionTrainingDataset(
        images_dir=dataset_files["images_dir"],
        captions_path=dataset_files["captions_path"],
        split_path=dataset_files["split_path"],
        split_name="train",
        transform=tensor_transform,
    )

    image, caption, image_name = dataset[1]

    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert image.dtype == torch.float32
    assert caption == "second train caption, with comma"
    assert image_name == TRAIN_IMAGE_1


def test_training_dataset_rejects_non_callable_transform(
    dataset_files: dict[str, Path],
) -> None:
    """A transform callable is required."""
    with pytest.raises(TypeError, match="transform must be callable"):
        ImageCaptionTrainingDataset(
            images_dir=dataset_files["images_dir"],
            captions_path=dataset_files["captions_path"],
            split_path=dataset_files["split_path"],
            split_name="train",
            transform=None,  # type: ignore[arg-type]
        )


def test_training_dataset_rejects_missing_images_directory(
    dataset_files: dict[str, Path],
    tmp_path: Path,
) -> None:
    """The image directory must exist."""
    with pytest.raises(FileNotFoundError):
        ImageCaptionTrainingDataset(
            images_dir=tmp_path / "missing-images",
            captions_path=dataset_files["captions_path"],
            split_path=dataset_files["split_path"],
            split_name="train",
            transform=tensor_transform,
        )


def test_training_dataset_rejects_file_as_images_directory(
    dataset_files: dict[str, Path],
) -> None:
    """images_dir must point to a directory."""
    with pytest.raises(NotADirectoryError):
        ImageCaptionTrainingDataset(
            images_dir=dataset_files["captions_path"],
            captions_path=dataset_files["captions_path"],
            split_path=dataset_files["split_path"],
            split_name="train",
            transform=tensor_transform,
        )


def test_dataset_rejects_empty_split(
    dataset_files: dict[str, Path],
) -> None:
    """An empty split must fail before training starts."""
    write_split_file(
        dataset_files["split_path"],
        train_images=[],
    )

    with pytest.raises(ValueError, match="split is empty"):
        ImageCaptionTrainingDataset(
            images_dir=dataset_files["images_dir"],
            captions_path=dataset_files["captions_path"],
            split_path=dataset_files["split_path"],
            split_name="train",
            transform=tensor_transform,
        )


def test_dataset_rejects_image_without_caption(
    dataset_files: dict[str, Path],
) -> None:
    """Every split image must have at least one caption."""
    uncaptured_image = "without_caption.jpg"
    write_image(dataset_files["images_dir"] / uncaptured_image)
    write_split_file(
        dataset_files["split_path"],
        train_images=[uncaptured_image],
    )

    with pytest.raises(ValueError, match="Images without captions"):
        ImageCaptionTrainingDataset(
            images_dir=dataset_files["images_dir"],
            captions_path=dataset_files["captions_path"],
            split_path=dataset_files["split_path"],
            split_name="train",
            transform=tensor_transform,
        )


def test_dataset_rejects_missing_image_file(
    dataset_files: dict[str, Path],
) -> None:
    """Every split filename must exist inside images_dir."""
    (dataset_files["images_dir"] / TRAIN_IMAGE_1).unlink()

    with pytest.raises(FileNotFoundError, match="Missing image files"):
        ImageCaptionTrainingDataset(
            images_dir=dataset_files["images_dir"],
            captions_path=dataset_files["captions_path"],
            split_path=dataset_files["split_path"],
            split_name="train",
            transform=tensor_transform,
        )


def test_evaluation_dataset_returns_each_image_once(
    dataset_files: dict[str, Path],
) -> None:
    """Evaluation dataset length must equal unique split images."""
    dataset = ImageCaptionEvaluationDataset(
        images_dir=dataset_files["images_dir"],
        captions_path=dataset_files["captions_path"],
        split_path=dataset_files["split_path"],
        split_name="test",
        transform=tensor_transform,
    )

    assert len(dataset) == 1

    image, references, image_name = dataset[0]

    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert references == [
        "test caption one",
        "test caption two",
    ]
    assert image_name == TEST_IMAGE


def test_evaluation_dataset_returns_reference_copy(
    dataset_files: dict[str, Path],
) -> None:
    """Mutating returned references must not alter stored captions."""
    dataset = ImageCaptionEvaluationDataset(
        images_dir=dataset_files["images_dir"],
        captions_path=dataset_files["captions_path"],
        split_path=dataset_files["split_path"],
        split_name="test",
        transform=tensor_transform,
    )

    _, references, _ = dataset[0]
    references.append("external mutation")

    assert dataset.captions_by_image[TEST_IMAGE] == [
        "test caption one",
        "test caption two",
    ]


def test_load_image_converts_grayscale_to_rgb(
    tmp_path: Path,
) -> None:
    """All source images must reach the transform in RGB mode."""
    image_path = tmp_path / "gray.png"
    write_image(image_path, mode="L")

    observed_modes: list[str] = []

    def capture_mode(image: Image.Image) -> Tensor:
        observed_modes.append(image.mode)
        return torch.zeros(
            3,
            image.height,
            image.width,
        )

    image_tensor = _load_image(
        image_path,
        capture_mode,
    )

    assert observed_modes == ["RGB"]
    assert image_tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_load_image_rejects_non_tensor_transform_output(
    tmp_path: Path,
) -> None:
    """Transforms must return torch.Tensor objects."""
    image_path = tmp_path / "image.jpg"
    write_image(image_path)

    def invalid_transform(image: Image.Image) -> str:
        del image
        return "not a tensor"

    with pytest.raises(TypeError, match="must return a torch.Tensor"):
        _load_image(
            image_path,
            invalid_transform,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "transform",
    [
        lambda image: torch.zeros(image.height, image.width),
        lambda image: torch.zeros(1, image.height, image.width),
        lambda image: torch.zeros(4, image.height, image.width),
    ],
)
def test_load_image_rejects_invalid_tensor_shape(
    tmp_path: Path,
    transform: Callable[[Image.Image], Tensor],
) -> None:
    """Transformed images must have shape (3, height, width)."""
    image_path = tmp_path / "image.jpg"
    write_image(image_path)

    with pytest.raises(ValueError, match="must have shape"):
        _load_image(image_path, transform)


def test_load_image_rejects_invalid_image_file(
    tmp_path: Path,
) -> None:
    """Corrupted image files must produce a clear ValueError."""
    image_path = tmp_path / "broken.jpg"
    image_path.write_text(
        "this is not an image",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid image file"):
        _load_image(image_path, tensor_transform)


def test_dataset_index_out_of_range_raises_index_error(
    dataset_files: dict[str, Path],
) -> None:
    """Dataset indexing follows normal Python sequence behavior."""
    dataset = ImageCaptionEvaluationDataset(
        images_dir=dataset_files["images_dir"],
        captions_path=dataset_files["captions_path"],
        split_path=dataset_files["split_path"],
        split_name="val",
        transform=tensor_transform,
    )

    with pytest.raises(IndexError):
        _ = dataset[10]