"""Tests for shared ViT and Swin image transformations."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from src.transforms import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IMAGE_STD,
    DEFAULT_RESIZE_SIZE,
    ImageTransforms,
    TransformSettings,
    create_evaluation_transform,
    create_image_transforms,
    create_training_transform,
    validate_transformed_image,
)


def _create_rgb_image(
    *,
    width: int = 320,
    height: int = 240,
    color: tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    return Image.new(
        mode="RGB",
        size=(width, height),
        color=color,
    )


def _create_horizontal_pattern_image() -> Image.Image:
    image = Image.new(
        mode="RGB",
        size=(224, 224),
        color=(0, 0, 0),
    )

    for x in range(112):
        for y in range(224):
            image.putpixel(
                (x, y),
                (255, 0, 0),
            )

    for x in range(112, 224):
        for y in range(224):
            image.putpixel(
                (x, y),
                (0, 0, 255),
            )

    return image


def _create_deterministic_training_settings(
    *,
    horizontal_flip_probability: float = 0.0,
) -> TransformSettings:
    return TransformSettings(
        image_size=224,
        resize_size=256,
        train_crop_scale=(1.0, 1.0),
        train_crop_ratio=(1.0, 1.0),
        horizontal_flip_probability=(
            horizontal_flip_probability
        ),
        brightness_jitter=0.0,
        contrast_jitter=0.0,
        saturation_jitter=0.0,
        hue_jitter=0.0,
    )


def test_default_constants_have_expected_values() -> None:
    assert DEFAULT_IMAGE_SIZE == 224
    assert DEFAULT_RESIZE_SIZE == 256
    assert DEFAULT_IMAGE_MEAN == (0.5, 0.5, 0.5)
    assert DEFAULT_IMAGE_STD == (0.5, 0.5, 0.5)


def test_transform_settings_have_expected_defaults() -> None:
    settings = TransformSettings()

    assert settings.image_size == 224
    assert settings.resize_size == 256
    assert settings.image_mean == (0.5, 0.5, 0.5)
    assert settings.image_std == (0.5, 0.5, 0.5)
    assert settings.train_crop_scale == (0.85, 1.0)
    assert settings.train_crop_ratio == (0.9, 1.1)
    assert settings.horizontal_flip_probability == 0.5
    assert settings.brightness_jitter == 0.1
    assert settings.contrast_jitter == 0.1
    assert settings.saturation_jitter == 0.1
    assert settings.hue_jitter == 0.02


def test_transform_settings_are_frozen() -> None:
    settings = TransformSettings()

    with pytest.raises(FrozenInstanceError):
        settings.image_size = 384  # type: ignore[misc]


def test_valid_default_settings_pass_validation() -> None:
    TransformSettings().validate()


def test_valid_custom_settings_pass_validation() -> None:
    settings = TransformSettings(
        image_size=384,
        resize_size=448,
        image_mean=(0.1, 0.2, 0.3),
        image_std=(0.4, 0.5, 0.6),
        train_crop_scale=(0.7, 1.0),
        train_crop_ratio=(0.8, 1.2),
        horizontal_flip_probability=0.25,
        brightness_jitter=0.2,
        contrast_jitter=0.3,
        saturation_jitter=0.4,
        hue_jitter=0.1,
    )

    settings.validate()


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("image_size", 0),
        ("image_size", -1),
        ("resize_size", 0),
        ("resize_size", -1),
    ],
)
def test_non_positive_integer_settings_are_rejected(
    field_name: str,
    field_value: int,
) -> None:
    settings = TransformSettings(
        **{field_name: field_value}
    )

    with pytest.raises(
        ValueError,
        match="must be greater than zero",
    ):
        settings.validate()


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("image_size", 224.0),
        ("image_size", True),
        ("image_size", "224"),
        ("resize_size", 256.0),
        ("resize_size", False),
        ("resize_size", "256"),
    ],
)
def test_non_integer_size_settings_are_rejected(
    field_name: str,
    field_value: object,
) -> None:
    settings = TransformSettings(
        **{field_name: field_value}  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        settings.validate()


def test_resize_size_smaller_than_image_size_is_rejected() -> None:
    settings = TransformSettings(
        image_size=256,
        resize_size=224,
    )

    with pytest.raises(
        ValueError,
        match="resize_size must be greater than or equal",
    ):
        settings.validate()


@pytest.mark.parametrize(
    "field_name",
    [
        "image_mean",
        "image_std",
    ],
)
def test_channel_values_must_be_tuples(
    field_name: str,
) -> None:
    settings = TransformSettings(
        **{
            field_name: [0.5, 0.5, 0.5],
        }  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError,
        match="must be a tuple",
    ):
        settings.validate()


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("image_mean", (0.5, 0.5)),
        ("image_mean", (0.5, 0.5, 0.5, 0.5)),
        ("image_std", (0.5, 0.5)),
        ("image_std", (0.5, 0.5, 0.5, 0.5)),
    ],
)
def test_channel_values_must_have_three_items(
    field_name: str,
    field_value: tuple[float, ...],
) -> None:
    settings = TransformSettings(
        **{field_name: field_value}  # type: ignore[arg-type]
    )

    with pytest.raises(
        ValueError,
        match="exactly three values",
    ):
        settings.validate()


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("image_mean", (0.5, "bad", 0.5)),
        ("image_mean", (0.5, True, 0.5)),
        ("image_std", (0.5, "bad", 0.5)),
        ("image_std", (0.5, False, 0.5)),
    ],
)
def test_channel_values_must_be_numeric(
    field_name: str,
    field_value: tuple[object, object, object],
) -> None:
    settings = TransformSettings(
        **{field_name: field_value}  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError,
        match="must be a number",
    ):
        settings.validate()


@pytest.mark.parametrize(
    "image_std",
    [
        (0.0, 0.5, 0.5),
        (-0.1, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, -1.0),
    ],
)
def test_image_standard_deviation_must_be_positive(
    image_std: tuple[float, float, float],
) -> None:
    settings = TransformSettings(
        image_std=image_std,
    )

    with pytest.raises(
        ValueError,
        match="must be greater than zero",
    ):
        settings.validate()


@pytest.mark.parametrize(
    "field_name",
    [
        "train_crop_scale",
        "train_crop_ratio",
    ],
)
def test_crop_ranges_must_be_tuples(
    field_name: str,
) -> None:
    settings = TransformSettings(
        **{
            field_name: [0.8, 1.0],
        }  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError,
        match="must be a tuple",
    ):
        settings.validate()


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("train_crop_scale", (0.8,)),
        ("train_crop_scale", (0.8, 0.9, 1.0)),
        ("train_crop_ratio", (0.8,)),
        ("train_crop_ratio", (0.8, 0.9, 1.0)),
    ],
)
def test_crop_ranges_must_have_two_values(
    field_name: str,
    field_value: tuple[float, ...],
) -> None:
    settings = TransformSettings(
        **{field_name: field_value}  # type: ignore[arg-type]
    )

    with pytest.raises(
        ValueError,
        match="exactly two values",
    ):
        settings.validate()


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("train_crop_scale", ("bad", 1.0)),
        ("train_crop_scale", (0.8, True)),
        ("train_crop_ratio", ("bad", 1.0)),
        ("train_crop_ratio", (0.8, False)),
    ],
)
def test_crop_range_values_must_be_numeric(
    field_name: str,
    field_value: tuple[object, object],
) -> None:
    settings = TransformSettings(
        **{field_name: field_value}  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError,
        match="must be a number",
    ):
        settings.validate()


@pytest.mark.parametrize(
    "train_crop_scale",
    [
        (0.0, 1.0),
        (-0.1, 1.0),
    ],
)
def test_crop_scale_lower_bound_must_be_positive(
    train_crop_scale: tuple[float, float],
) -> None:
    settings = TransformSettings(
        train_crop_scale=train_crop_scale,
    )

    with pytest.raises(
        ValueError,
        match="must be greater than",
    ):
        settings.validate()


@pytest.mark.parametrize(
    "train_crop_scale",
    [
        (0.8, 1.1),
        (1.1, 1.2),
    ],
)
def test_crop_scale_upper_bound_cannot_exceed_one(
    train_crop_scale: tuple[float, float],
) -> None:
    settings = TransformSettings(
        train_crop_scale=train_crop_scale,
    )

    with pytest.raises(
        ValueError,
        match="less than or equal to 1.0",
    ):
        settings.validate()


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("train_crop_scale", (0.9, 0.8)),
        ("train_crop_ratio", (1.1, 0.9)),
    ],
)
def test_crop_range_upper_bound_cannot_be_smaller(
    field_name: str,
    field_value: tuple[float, float],
) -> None:
    settings = TransformSettings(
        **{field_name: field_value}
    )

    with pytest.raises(
        ValueError,
        match="greater than or equal",
    ):
        settings.validate()


@pytest.mark.parametrize(
    "train_crop_ratio",
    [
        (0.0, 1.0),
        (-0.5, 1.0),
    ],
)
def test_crop_ratio_lower_bound_must_be_positive(
    train_crop_ratio: tuple[float, float],
) -> None:
    settings = TransformSettings(
        train_crop_ratio=train_crop_ratio,
    )

    with pytest.raises(
        ValueError,
        match="must be greater than",
    ):
        settings.validate()


@pytest.mark.parametrize(
    "horizontal_flip_probability",
    [
        -0.1,
        1.1,
    ],
)
def test_flip_probability_must_be_between_zero_and_one(
    horizontal_flip_probability: float,
) -> None:
    settings = TransformSettings(
        horizontal_flip_probability=(
            horizontal_flip_probability
        ),
    )

    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        settings.validate()


@pytest.mark.parametrize(
    "horizontal_flip_probability",
    [
        True,
        "0.5",
        None,
    ],
)
def test_flip_probability_must_be_numeric(
    horizontal_flip_probability: object,
) -> None:
    settings = TransformSettings(
        horizontal_flip_probability=(
            horizontal_flip_probability
        ),  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError,
        match="must be a number",
    ):
        settings.validate()


@pytest.mark.parametrize(
    "field_name",
    [
        "brightness_jitter",
        "contrast_jitter",
        "saturation_jitter",
    ],
)
@pytest.mark.parametrize(
    "field_value",
    [
        -0.1,
        -1.0,
    ],
)
def test_non_hue_jitter_values_cannot_be_negative(
    field_name: str,
    field_value: float,
) -> None:
    settings = TransformSettings(
        **{field_name: field_value}
    )

    with pytest.raises(
        ValueError,
        match="zero or greater",
    ):
        settings.validate()


@pytest.mark.parametrize(
    "field_name",
    [
        "brightness_jitter",
        "contrast_jitter",
        "saturation_jitter",
    ],
)
@pytest.mark.parametrize(
    "field_value",
    [
        True,
        "0.1",
        None,
    ],
)
def test_non_hue_jitter_values_must_be_numeric(
    field_name: str,
    field_value: object,
) -> None:
    settings = TransformSettings(
        **{field_name: field_value}  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError,
        match="must be a number",
    ):
        settings.validate()


@pytest.mark.parametrize(
    "hue_jitter",
    [
        -0.1,
        0.51,
        1.0,
    ],
)
def test_hue_jitter_must_be_in_supported_range(
    hue_jitter: float,
) -> None:
    settings = TransformSettings(
        hue_jitter=hue_jitter,
    )

    with pytest.raises(
        ValueError,
        match="between 0.0 and 0.5",
    ):
        settings.validate()


@pytest.mark.parametrize(
    "hue_jitter",
    [
        True,
        "0.1",
        None,
    ],
)
def test_hue_jitter_must_be_numeric(
    hue_jitter: object,
) -> None:
    settings = TransformSettings(
        hue_jitter=hue_jitter,  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError,
        match="must be a number",
    ):
        settings.validate()


def test_training_transform_has_expected_operations() -> None:
    transform = create_training_transform()

    assert isinstance(transform, Compose)
    assert len(transform.transforms) == 5
    assert isinstance(
        transform.transforms[0],
        RandomResizedCrop,
    )
    assert isinstance(
        transform.transforms[1],
        RandomHorizontalFlip,
    )
    assert isinstance(
        transform.transforms[2],
        ColorJitter,
    )
    assert isinstance(
        transform.transforms[3],
        ToTensor,
    )
    assert isinstance(
        transform.transforms[4],
        Normalize,
    )


def test_evaluation_transform_has_expected_operations() -> None:
    transform = create_evaluation_transform()

    assert isinstance(transform, Compose)
    assert len(transform.transforms) == 4
    assert isinstance(
        transform.transforms[0],
        Resize,
    )
    assert isinstance(
        transform.transforms[1],
        CenterCrop,
    )
    assert isinstance(
        transform.transforms[2],
        ToTensor,
    )
    assert isinstance(
        transform.transforms[3],
        Normalize,
    )


def test_create_image_transforms_returns_expected_container() -> None:
    transforms = create_image_transforms()

    assert isinstance(
        transforms,
        ImageTransforms,
    )
    assert isinstance(
        transforms.train,
        Compose,
    )
    assert isinstance(
        transforms.evaluation,
        Compose,
    )


def test_image_transforms_container_is_frozen() -> None:
    transforms = create_image_transforms()

    with pytest.raises(FrozenInstanceError):
        transforms.train = Compose([])  # type: ignore[misc]


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (224, 224),
        (320, 240),
        (240, 320),
        (640, 480),
        (480, 640),
    ],
)
def test_training_transform_returns_expected_shape(
    width: int,
    height: int,
) -> None:
    image = _create_rgb_image(
        width=width,
        height=height,
    )

    transformed = create_training_transform()(image)

    assert isinstance(transformed, Tensor)
    assert transformed.shape == (
        3,
        224,
        224,
    )
    assert transformed.dtype == torch.float32


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (224, 224),
        (320, 240),
        (240, 320),
        (640, 480),
        (480, 640),
    ],
)
def test_evaluation_transform_returns_expected_shape(
    width: int,
    height: int,
) -> None:
    image = _create_rgb_image(
        width=width,
        height=height,
    )

    transformed = create_evaluation_transform()(image)

    assert isinstance(transformed, Tensor)
    assert transformed.shape == (
        3,
        224,
        224,
    )
    assert transformed.dtype == torch.float32


def test_custom_image_size_changes_training_output_shape() -> None:
    settings = TransformSettings(
        image_size=384,
        resize_size=448,
    )
    image = _create_rgb_image(
        width=500,
        height=420,
    )

    transformed = create_training_transform(
        settings
    )(image)

    assert transformed.shape == (
        3,
        384,
        384,
    )


def test_custom_image_size_changes_evaluation_output_shape() -> None:
    settings = TransformSettings(
        image_size=384,
        resize_size=448,
    )
    image = _create_rgb_image(
        width=500,
        height=420,
    )

    transformed = create_evaluation_transform(
        settings
    )(image)

    assert transformed.shape == (
        3,
        384,
        384,
    )


def test_black_image_is_normalized_to_negative_one() -> None:
    settings = _create_deterministic_training_settings()
    image = _create_rgb_image(
        width=224,
        height=224,
        color=(0, 0, 0),
    )

    transformed = create_training_transform(
        settings
    )(image)

    assert torch.allclose(
        transformed,
        torch.full_like(
            transformed,
            -1.0,
        ),
    )


def test_white_image_is_normalized_to_positive_one() -> None:
    settings = _create_deterministic_training_settings()
    image = _create_rgb_image(
        width=224,
        height=224,
        color=(255, 255, 255),
    )

    transformed = create_training_transform(
        settings
    )(image)

    assert torch.allclose(
        transformed,
        torch.full_like(
            transformed,
            1.0,
        ),
    )


def test_custom_normalization_values_are_applied() -> None:
    settings = TransformSettings(
        image_size=224,
        resize_size=224,
        image_mean=(0.0, 0.0, 0.0),
        image_std=(1.0, 1.0, 1.0),
        train_crop_scale=(1.0, 1.0),
        train_crop_ratio=(1.0, 1.0),
        horizontal_flip_probability=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
        saturation_jitter=0.0,
        hue_jitter=0.0,
    )
    image = _create_rgb_image(
        width=224,
        height=224,
        color=(255, 255, 255),
    )

    transformed = create_training_transform(
        settings
    )(image)

    assert torch.allclose(
        transformed,
        torch.ones_like(transformed),
    )


def test_evaluation_transform_is_deterministic() -> None:
    transform = create_evaluation_transform()
    image = _create_rgb_image(
        width=320,
        height=240,
        color=(50, 100, 150),
    )

    first_output = transform(image)
    second_output = transform(image)

    assert torch.equal(
        first_output,
        second_output,
    )


def test_training_transform_without_randomness_is_deterministic() -> None:
    settings = _create_deterministic_training_settings()
    transform = create_training_transform(settings)
    image = _create_rgb_image(
        width=224,
        height=224,
        color=(50, 100, 150),
    )

    first_output = transform(image)
    second_output = transform(image)

    assert torch.equal(
        first_output,
        second_output,
    )


def test_horizontal_flip_probability_zero_preserves_pattern() -> None:
    settings = _create_deterministic_training_settings(
        horizontal_flip_probability=0.0,
    )
    transform = create_training_transform(settings)
    image = _create_horizontal_pattern_image()

    transformed = transform(image)

    left_red_mean = transformed[0, :, :112].mean()
    right_red_mean = transformed[0, :, 112:].mean()

    assert left_red_mean > right_red_mean


def test_horizontal_flip_probability_one_flips_pattern() -> None:
    settings = _create_deterministic_training_settings(
        horizontal_flip_probability=1.0,
    )
    transform = create_training_transform(settings)
    image = _create_horizontal_pattern_image()

    transformed = transform(image)

    left_red_mean = transformed[0, :, :112].mean()
    right_red_mean = transformed[0, :, 112:].mean()

    assert left_red_mean < right_red_mean


def test_create_training_transform_validates_settings() -> None:
    settings = TransformSettings(
        image_size=0,
    )

    with pytest.raises(ValueError):
        create_training_transform(settings)


def test_create_evaluation_transform_validates_settings() -> None:
    settings = TransformSettings(
        resize_size=0,
    )

    with pytest.raises(ValueError):
        create_evaluation_transform(settings)


def test_create_image_transforms_validates_settings() -> None:
    settings = TransformSettings(
        horizontal_flip_probability=2.0,
    )

    with pytest.raises(ValueError):
        create_image_transforms(settings)


def test_validate_transformed_image_accepts_valid_tensor() -> None:
    image = torch.zeros(
        3,
        224,
        224,
        dtype=torch.float32,
    )

    validate_transformed_image(image)


def test_validate_transformed_image_accepts_custom_size() -> None:
    image = torch.zeros(
        3,
        384,
        384,
        dtype=torch.float32,
    )

    validate_transformed_image(
        image,
        expected_size=384,
    )


@pytest.mark.parametrize(
    "image",
    [
        None,
        "image",
        [[0.0]],
        _create_rgb_image(),
    ],
)
def test_validate_transformed_image_rejects_non_tensor(
    image: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be a torch.Tensor",
    ):
        validate_transformed_image(
            image,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "shape",
    [
        (224, 224),
        (1, 224, 224),
        (4, 224, 224),
        (3, 256, 224),
        (3, 224, 256),
        (2, 3, 224, 224),
    ],
)
def test_validate_transformed_image_rejects_wrong_shape(
    shape: tuple[int, ...],
) -> None:
    image = torch.zeros(
        *shape,
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="must have shape",
    ):
        validate_transformed_image(image)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    ],
)
def test_validate_transformed_image_rejects_non_float_dtype(
    dtype: torch.dtype,
) -> None:
    image = torch.zeros(
        3,
        224,
        224,
        dtype=dtype,
    )

    with pytest.raises(
        TypeError,
        match="floating-point dtype",
    ):
        validate_transformed_image(image)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.float64,
    ],
)
def test_validate_transformed_image_accepts_float_dtypes(
    dtype: torch.dtype,
) -> None:
    image = torch.zeros(
        3,
        224,
        224,
        dtype=dtype,
    )

    validate_transformed_image(image)


@pytest.mark.parametrize(
    "expected_size",
    [
        0,
        -1,
    ],
)
def test_validate_transformed_image_rejects_non_positive_size(
    expected_size: int,
) -> None:
    image = torch.zeros(
        3,
        224,
        224,
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="must be greater than zero",
    ):
        validate_transformed_image(
            image,
            expected_size=expected_size,
        )


@pytest.mark.parametrize(
    "expected_size",
    [
        224.0,
        True,
        "224",
    ],
)
def test_validate_transformed_image_rejects_non_integer_size(
    expected_size: object,
) -> None:
    image = torch.zeros(
        3,
        224,
        224,
        dtype=torch.float32,
    )

    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        validate_transformed_image(
            image,
            expected_size=expected_size,  # type: ignore[arg-type]
        )