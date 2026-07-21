"""Datasets for training and evaluating image-captioning models.

Splits are applied at the image level, so every caption belonging to an
image remains in the same train, validation, or test split.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable, TypeAlias

from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset


TRAIN_IMAGES_KEY = "train_images"
VAL_IMAGES_KEY = "val_images"
TEST_IMAGES_KEY = "test_images"

SPLIT_TO_KEY = {
    "train": TRAIN_IMAGES_KEY,
    "val": VAL_IMAGES_KEY,
    "test": TEST_IMAGES_KEY,
}

ImageTransform: TypeAlias = Callable[[Image.Image], Tensor]
CaptionsByImage: TypeAlias = dict[str, list[str]]
TrainingSample: TypeAlias = tuple[Tensor, str, str]
EvaluationSample: TypeAlias = tuple[Tensor, list[str], str]
TrainingRecord: TypeAlias = tuple[str, str]


def _require_file(path: str | Path, name: str) -> Path:
    """Return a validated regular-file path."""
    resolved = Path(path)

    if not resolved.exists():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file: {resolved}")

    return resolved


def _require_directory(path: str | Path, name: str) -> Path:
    """Return a validated directory path."""
    resolved = Path(path)

    if not resolved.exists():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"{name} must be a directory: {resolved}")

    return resolved


def _parse_caption_row(row: str, line_number: int) -> tuple[str, str]:
    """Parse either ``image,caption`` or ``image.jpg#0<TAB>caption``."""
    if "\t" in row:
        image_field, caption = row.split("\t", maxsplit=1)
        image_name = image_field.split("#", maxsplit=1)[0]
    elif "," in row:
        image_name, caption = row.split(",", maxsplit=1)
    else:
        raise ValueError(
            f"Invalid caption row at line {line_number}. Expected "
            "'image,caption' or 'image.jpg#0<TAB>caption'."
        )

    image_name = image_name.strip()
    caption = caption.strip()

    if not image_name:
        raise ValueError(f"Missing image name at line {line_number}.")
    if not caption:
        raise ValueError(
            f"Missing caption at line {line_number} for {image_name!r}."
        )

    return image_name, caption


def read_captions_file(
    captions_path: str | Path,
) -> CaptionsByImage:
    """Read Flickr8k captions and group them by image filename."""
    path = _require_file(captions_path, "captions_path")
    captions_by_image: CaptionsByImage = {}

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, raw_row in enumerate(file, start=1):
            row = raw_row.strip()

            if not row:
                continue

            normalized = row.lower().replace(" ", "")
            if line_number == 1 and normalized in {
                "image,caption",
                "image_name,caption",
                "filename,caption",
            }:
                continue

            image_name, caption = _parse_caption_row(row, line_number)
            captions_by_image.setdefault(image_name, []).append(caption)

    if not captions_by_image:
        raise ValueError(f"No valid captions were found in {path}.")

    return captions_by_image


def _validate_split_list(value: object, split_key: str) -> list[str]:
    """Validate one image-name list from shared_split.json."""
    if not isinstance(value, list):
        raise TypeError(
            f"{split_key!r} must contain a list, "
            f"not {type(value).__name__}."
        )

    image_names: list[str] = []

    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(
                f"{split_key}[{index}] must be a string, "
                f"not {type(item).__name__}."
            )

        image_name = item.strip()
        if not image_name:
            raise ValueError(f"{split_key}[{index}] cannot be empty.")

        image_names.append(image_name)

    duplicates = sorted(
        name for name, count in Counter(image_names).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            f"Duplicate image names in {split_key!r}: {duplicates[:10]}"
        )

    return image_names


def read_shared_split(
    split_path: str | Path,
) -> dict[str, list[str]]:
    """Read all splits and verify that image sets do not overlap."""
    path = _require_file(split_path, "split_path")

    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in split file: {path}") from error

    if not isinstance(payload, dict):
        raise ValueError("The split JSON root must be an object.")

    splits: dict[str, list[str]] = {}

    for split_name, split_key in SPLIT_TO_KEY.items():
        if split_key not in payload:
            raise KeyError(f"Missing split key: {split_key!r}")

        splits[split_name] = _validate_split_list(
            payload[split_key],
            split_key,
        )

    split_sets = {name: set(values) for name, values in splits.items()}
    overlap_pairs = (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    )

    for left_name, right_name in overlap_pairs:
        overlap = split_sets[left_name] & split_sets[right_name]
        if overlap:
            raise ValueError(
                "Data leakage detected between "
                f"{left_name!r} and {right_name!r}: "
                f"{sorted(overlap)[:10]}"
            )

    return splits


def load_split_image_names(
    split_path: str | Path,
    split_name: str,
) -> list[str]:
    """Load image filenames belonging to ``train``, ``val``, or ``test``."""
    if not isinstance(split_name, str):
        raise TypeError(
            "split_name must be a string, "
            f"not {type(split_name).__name__}."
        )

    normalized = split_name.strip().lower()
    if normalized not in SPLIT_TO_KEY:
        raise ValueError(
            "split_name must be 'train', 'val', or 'test', "
            f"not {split_name!r}."
        )

    return list(read_shared_split(split_path)[normalized])


def _validate_split_content(
    images_dir: Path,
    image_names: list[str],
    captions: CaptionsByImage,
    split_name: str,
) -> None:
    """Fail early when a split contains missing files or captions."""
    if not image_names:
        raise ValueError(f"The {split_name!r} split is empty.")

    without_captions = [
        name for name in image_names if not captions.get(name)
    ]
    if without_captions:
        raise ValueError(
            f"Images without captions in {split_name!r}: "
            f"{without_captions[:10]}"
        )

    missing_files = [
        name for name in image_names if not (images_dir / name).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Missing image files in {split_name!r}: {missing_files[:10]}"
        )


def _load_image(
    image_path: Path,
    transform: ImageTransform,
) -> Tensor:
    """Open one RGB image and return its transformed Tensor."""
    try:
        with Image.open(image_path) as image:
            image_tensor = transform(image.convert("RGB"))
    except UnidentifiedImageError as error:
        raise ValueError(f"Invalid image file: {image_path}") from error
    except OSError as error:
        raise OSError(f"Could not read image: {image_path}") from error

    if not isinstance(image_tensor, Tensor):
        raise TypeError(
            "transform must return a torch.Tensor, "
            f"not {type(image_tensor).__name__}."
        )
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError(
            "The transformed image must have shape "
            f"(3, height, width), not {tuple(image_tensor.shape)}."
        )

    return image_tensor


class ImageCaptionTrainingDataset(Dataset[TrainingSample]):
    """Return one image-caption pair per sample."""

    def __init__(
        self,
        images_dir: str | Path,
        captions_path: str | Path,
        split_path: str | Path,
        split_name: str,
        transform: ImageTransform,
    ) -> None:
        if not callable(transform):
            raise TypeError("transform must be callable.")

        self.images_dir = _require_directory(images_dir, "images_dir")
        self.transform = transform
        self.captions_by_image = read_captions_file(captions_path)
        self.image_names = load_split_image_names(split_path, split_name)
        self.split_name = split_name.strip().lower()

        _validate_split_content(
            self.images_dir,
            self.image_names,
            self.captions_by_image,
            self.split_name,
        )

        self.samples: list[TrainingRecord] = [
            (image_name, caption)
            for image_name in self.image_names
            for caption in self.captions_by_image[image_name]
        ]

    def __len__(self) -> int:
        """Return the number of image-caption pairs."""
        return len(self.samples)

    def __getitem__(self, index: int) -> TrainingSample:
        """Load one transformed image-caption pair."""
        image_name, caption = self.samples[index]
        image_tensor = _load_image(
            self.images_dir / image_name,
            self.transform,
        )
        return image_tensor, caption, image_name


class ImageCaptionEvaluationDataset(Dataset[EvaluationSample]):
    """Return every image once together with all reference captions."""

    def __init__(
        self,
        images_dir: str | Path,
        captions_path: str | Path,
        split_path: str | Path,
        split_name: str,
        transform: ImageTransform,
    ) -> None:
        if not callable(transform):
            raise TypeError("transform must be callable.")

        self.images_dir = _require_directory(images_dir, "images_dir")
        self.transform = transform
        self.captions_by_image = read_captions_file(captions_path)
        self.image_names = load_split_image_names(split_path, split_name)
        self.split_name = split_name.strip().lower()

        _validate_split_content(
            self.images_dir,
            self.image_names,
            self.captions_by_image,
            self.split_name,
        )

    def __len__(self) -> int:
        """Return the number of unique images."""
        return len(self.image_names)

    def __getitem__(self, index: int) -> EvaluationSample:
        """Load one image and a copy of all its reference captions."""
        image_name = self.image_names[index]
        image_tensor = _load_image(
            self.images_dir / image_name,
            self.transform,
        )
        reference_captions = list(self.captions_by_image[image_name])
        return image_tensor, reference_captions, image_name