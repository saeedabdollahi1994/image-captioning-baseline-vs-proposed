"""
Datasets for training and evaluating image-captioning models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, TypeAlias

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.common.constants import (
    TEST_IMAGES_KEY,
    TRAIN_IMAGES_KEY,
    VAL_IMAGES_KEY,
)


ImageTransform: TypeAlias = Callable[[Image.Image], Tensor]
CaptionsByImage: TypeAlias = dict[str, list[str]]
TrainingSample: TypeAlias = tuple[Tensor, str, str]
EvaluationSample: TypeAlias = tuple[Tensor, list[str], str]


def read_captions_file(
    captions_path: str | Path,
) -> CaptionsByImage:
    """
    Read Flickr8k captions and group them by image name.

    Args:
        captions_path: Path to the Flickr8k captions file.

    Returns:
        Mapping from each image filename to its reference captions.

    Raises:
        FileNotFoundError: If the captions file does not exist.
        ValueError: If a row has an invalid format.
    """
    # TODO: Convert captions_path to Path.

    # TODO: Verify that the file exists and is a regular file.

    # TODO: Read the file using UTF-8 encoding.

    # TODO: Skip the header row if one exists.

    # TODO: Split each row into image name and caption.

    # TODO: Remove extra whitespace from captions.

    # TODO: Group all captions belonging to the same image.

    raise NotImplementedError


def load_split_image_names(
    split_path: str | Path,
    split_name: str,
) -> list[str]:
    """
    Load image names belonging to one dataset split.

    Args:
        split_path: Path to shared_split.json.
        split_name: One of ``train``, ``val``, or ``test``.

    Returns:
        Image filenames belonging to the requested split.

    Raises:
        FileNotFoundError: If the split file does not exist.
        ValueError: If split_name is unsupported.
        KeyError: If the expected split key is missing.
    """
    # TODO: Map split names to their JSON keys:
    #       train -> TRAIN_IMAGES_KEY
    #       val   -> VAL_IMAGES_KEY
    #       test  -> TEST_IMAGES_KEY

    # TODO: Validate split_name.

    # TODO: Read shared_split.json.

    # TODO: Extract and return a copy of the requested image list.

    raise NotImplementedError


class ImageCaptionTrainingDataset(Dataset[TrainingSample]):
    """
    Dataset that returns one image-caption pair per sample.

    Each image may appear multiple times because Flickr8k provides
    multiple reference captions for every image.
    """

    def __init__(
        self,
        images_dir: str | Path,
        captions_path: str | Path,
        split_path: str | Path,
        split_name: str,
        transform: ImageTransform,
    ) -> None:
        """
        Initialize the training dataset.

        Args:
            images_dir: Directory containing Flickr8k images.
            captions_path: Path to captions.txt.
            split_path: Path to shared_split.json.
            split_name: Usually ``train`` or ``val``.
            transform: Image preprocessing and augmentation function.
        """
        # TODO: Validate and store images_dir.

        # TODO: Store transform.

        # TODO: Read all captions using read_captions_file().

        # TODO: Load split image names using load_split_image_names().

        # TODO: Build self.samples as:
        #       list[tuple[image_name, caption]]

        # TODO: Raise an error if an image has no caption.

    def __len__(self) -> int:
        """
        Return the number of image-caption pairs.
        """
        # TODO: Return the number of entries in self.samples.

        raise NotImplementedError

    def __getitem__(
        self,
        index: int,
    ) -> TrainingSample:
        """
        Load one transformed image-caption pair.

        Args:
            index: Position of the sample.

        Returns:
            Tuple containing image tensor, raw caption, and image filename.
        """
        # TODO: Read image_name and caption from self.samples.

        # TODO: Construct the complete image path.

        # TODO: Open the image and convert it to RGB.

        # TODO: Apply the image transform.

        # TODO: Return:
        #       image_tensor, caption, image_name

        raise NotImplementedError


class ImageCaptionEvaluationDataset(Dataset[EvaluationSample]):
    """
    Dataset that returns one image and all its reference captions.

    Every image appears exactly once, making this dataset suitable
    for validation, testing, and caption-metric calculation.
    """

    def __init__(
        self,
        images_dir: str | Path,
        captions_path: str | Path,
        split_path: str | Path,
        split_name: str,
        transform: ImageTransform,
    ) -> None:
        """
        Initialize the evaluation dataset.

        Args:
            images_dir: Directory containing Flickr8k images.
            captions_path: Path to captions.txt.
            split_path: Path to shared_split.json.
            split_name: Usually ``val`` or ``test``.
            transform: Image preprocessing function.
        """
        # TODO: Validate and store images_dir.

        # TODO: Store transform.

        # TODO: Read all captions using read_captions_file().

        # TODO: Load and store the requested image names.

        # TODO: Verify that every image has at least one reference caption.

    def __len__(self) -> int:
        """
        Return the number of unique images.
        """
        # TODO: Return the number of image names.

        raise NotImplementedError

    def __getitem__(
        self,
        index: int,
    ) -> EvaluationSample:
        """
        Load one image and all its reference captions.

        Args:
            index: Position of the image.

        Returns:
            Tuple containing image tensor, reference captions,
            and image filename.
        """
        # TODO: Read the image name at the requested index.

        # TODO: Construct the complete image path.

        # TODO: Open the image and convert it to RGB.

        # TODO: Apply the image transform.

        # TODO: Retrieve all reference captions for the image.

        # TODO: Return:
        #       image_tensor, reference_captions, image_name

        raise NotImplementedError