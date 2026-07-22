"""Image transformations shared by ViT and Swin encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from torch import Tensor
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


ChannelValues: TypeAlias = tuple[float, float, float]

DEFAULT_IMAGE_SIZE = 224
DEFAULT_RESIZE_SIZE = 256
DEFAULT_IMAGE_MEAN: ChannelValues = (0.5, 0.5, 0.5)
DEFAULT_IMAGE_STD: ChannelValues = (0.5, 0.5, 0.5)


@dataclass(frozen=True, slots=True)
class TransformSettings:
    """Configuration for training and evaluation image transforms."""

    image_size: int = DEFAULT_IMAGE_SIZE
    resize_size: int = DEFAULT_RESIZE_SIZE
    image_mean: ChannelValues = DEFAULT_IMAGE_MEAN
    image_std: ChannelValues = DEFAULT_IMAGE_STD
    train_crop_scale: tuple[float, float] = (0.85, 1.0)
    train_crop_ratio: tuple[float, float] = (0.9, 1.1)
    horizontal_flip_probability: float = 0.5
    brightness_jitter: float = 0.1
    contrast_jitter: float = 0.1
    saturation_jitter: float = 0.1
    hue_jitter: float = 0.02

    def validate(self) -> None:
        """Validate all transform settings."""
        _validate_positive_integer(
            self.image_size,
            "image_size",
        )
        _validate_positive_integer(
            self.resize_size,
            "resize_size",
        )

        if self.resize_size < self.image_size:
            raise ValueError(
                "resize_size must be greater than or equal to "
                f"image_size, not {self.resize_size} < "
                f"{self.image_size}."
            )

        _validate_channel_values(
            self.image_mean,
            "image_mean",
            require_positive=False,
        )
        _validate_channel_values(
            self.image_std,
            "image_std",
            require_positive=True,
        )
        _validate_range(
            self.train_crop_scale,
            "train_crop_scale",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
        _validate_range(
            self.train_crop_ratio,
            "train_crop_ratio",
            minimum=0.0,
            maximum=None,
            minimum_inclusive=False,
        )
        _validate_probability(
            self.horizontal_flip_probability,
            "horizontal_flip_probability",
        )

        _validate_non_negative_float(
            self.brightness_jitter,
            "brightness_jitter",
        )
        _validate_non_negative_float(
            self.contrast_jitter,
            "contrast_jitter",
        )
        _validate_non_negative_float(
            self.saturation_jitter,
            "saturation_jitter",
        )

        if isinstance(self.hue_jitter, bool) or not isinstance(
            self.hue_jitter,
            (int, float),
        ):
            raise TypeError(
                "hue_jitter must be a number, "
                f"not {type(self.hue_jitter).__name__}."
            )

        if not 0.0 <= float(self.hue_jitter) <= 0.5:
            raise ValueError(
                "hue_jitter must be between 0.0 and 0.5, "
                f"not {self.hue_jitter}."
            )


@dataclass(frozen=True, slots=True)
class ImageTransforms:
    """Training and evaluation transform pipelines."""

    train: Compose
    evaluation: Compose


def _validate_positive_integer(
    value: int,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name} must be an integer, "
            f"not {type(value).__name__}."
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than zero, not {value}."
        )


def _validate_channel_values(
    values: ChannelValues,
    name: str,
    *,
    require_positive: bool,
) -> None:
    if not isinstance(values, tuple):
        raise TypeError(
            f"{name} must be a tuple, "
            f"not {type(values).__name__}."
        )

    if len(values) != 3:
        raise ValueError(
            f"{name} must contain exactly three values, "
            f"not {len(values)}."
        )

    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name}[{index}] must be a number, "
                f"not {type(value).__name__}."
            )

        if require_positive and float(value) <= 0.0:
            raise ValueError(
                f"{name}[{index}] must be greater than zero, "
                f"not {value}."
            )


def _validate_range(
    values: tuple[float, float],
    name: str,
    *,
    minimum: float,
    maximum: float | None,
    minimum_inclusive: bool,
) -> None:
    if not isinstance(values, tuple):
        raise TypeError(
            f"{name} must be a tuple, "
            f"not {type(values).__name__}."
        )

    if len(values) != 2:
        raise ValueError(
            f"{name} must contain exactly two values, "
            f"not {len(values)}."
        )

    lower, upper = values

    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name}[{index}] must be a number, "
                f"not {type(value).__name__}."
            )

    if minimum_inclusive:
        lower_is_valid = float(lower) >= minimum
    else:
        lower_is_valid = float(lower) > minimum

    if not lower_is_valid:
        comparison = "greater than or equal to" if (
            minimum_inclusive
        ) else "greater than"
        raise ValueError(
            f"{name}[0] must be {comparison} {minimum}, "
            f"not {lower}."
        )

    if float(upper) < float(lower):
        raise ValueError(
            f"{name}[1] must be greater than or equal to "
            f"{name}[0], not {upper} < {lower}."
        )

    if maximum is not None and float(upper) > maximum:
        raise ValueError(
            f"{name}[1] must be less than or equal to "
            f"{maximum}, not {upper}."
        )


def _validate_probability(
    value: float,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{name} must be a number, "
            f"not {type(value).__name__}."
        )

    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(
            f"{name} must be between 0.0 and 1.0, "
            f"not {value}."
        )


def _validate_non_negative_float(
    value: float,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{name} must be a number, "
            f"not {type(value).__name__}."
        )

    if float(value) < 0.0:
        raise ValueError(
            f"{name} must be zero or greater, not {value}."
        )


def create_training_transform(
    settings: TransformSettings | None = None,
) -> Compose:
    """Create the stochastic image transform used for training."""
    resolved_settings = settings or TransformSettings()
    resolved_settings.validate()

    return Compose(
        [
            RandomResizedCrop(
                size=resolved_settings.image_size,
                scale=resolved_settings.train_crop_scale,
                ratio=resolved_settings.train_crop_ratio,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            RandomHorizontalFlip(
                p=resolved_settings.horizontal_flip_probability,
            ),
            ColorJitter(
                brightness=resolved_settings.brightness_jitter,
                contrast=resolved_settings.contrast_jitter,
                saturation=resolved_settings.saturation_jitter,
                hue=resolved_settings.hue_jitter,
            ),
            ToTensor(),
            Normalize(
                mean=resolved_settings.image_mean,
                std=resolved_settings.image_std,
            ),
        ]
    )


def create_evaluation_transform(
    settings: TransformSettings | None = None,
) -> Compose:
    """Create the deterministic transform used for validation and test."""
    resolved_settings = settings or TransformSettings()
    resolved_settings.validate()

    return Compose(
        [
            Resize(
                size=resolved_settings.resize_size,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            CenterCrop(
                size=resolved_settings.image_size,
            ),
            ToTensor(),
            Normalize(
                mean=resolved_settings.image_mean,
                std=resolved_settings.image_std,
            ),
        ]
    )


def create_image_transforms(
    settings: TransformSettings | None = None,
) -> ImageTransforms:
    """Create both training and evaluation image transforms."""
    resolved_settings = settings or TransformSettings()
    resolved_settings.validate()

    return ImageTransforms(
        train=create_training_transform(resolved_settings),
        evaluation=create_evaluation_transform(
            resolved_settings
        ),
    )


def validate_transformed_image(
    image: Tensor,
    *,
    expected_size: int = DEFAULT_IMAGE_SIZE,
) -> None:
    """Validate a transformed image before it enters a data loader."""
    _validate_positive_integer(
        expected_size,
        "expected_size",
    )

    if not isinstance(image, Tensor):
        raise TypeError(
            "image must be a torch.Tensor, "
            f"not {type(image).__name__}."
        )

    expected_shape = (
        3,
        expected_size,
        expected_size,
    )

    if tuple(image.shape) != expected_shape:
        raise ValueError(
            f"Transformed image must have shape "
            f"{expected_shape}, not {tuple(image.shape)}."
        )

    if not image.is_floating_point():
        raise TypeError(
            "Transformed image must have a floating-point "
            f"dtype, not {image.dtype}."
        )