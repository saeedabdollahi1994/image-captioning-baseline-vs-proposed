"""
Tests for the vision encoders.

Run this file from the project root with:

    pytest tests/test_encoders.py -v
"""

import pytest
import torch
from torch import Tensor

from src.proposed.encoders import (
    BaseVisionEncoder,
    DualVisionEncoder,
    SwinEncoder,
    ViTEncoder,
)


BATCH_SIZE = 1
IMAGE_CHANNELS = 3
IMAGE_SIZE = 224


@pytest.fixture(scope="session")
def sample_images() -> Tensor:
    """
    Create a fake image batch for testing.

    Returns:
        Tensor with shape:

            (batch_size, channels, height, width)
    """
    torch.manual_seed(42)

    return torch.randn(
        BATCH_SIZE,
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )


@pytest.fixture(scope="session")
def dual_encoder() -> DualVisionEncoder:
    """
    Create one frozen dual encoder for all tests.

    Session scope prevents the pretrained models from being
    loaded again for every test function.
    """
    encoder = DualVisionEncoder(
        freeze_encoders=True,
    )

    encoder.eval()

    return encoder


@pytest.fixture(scope="session")
def encoder_outputs(
    dual_encoder: DualVisionEncoder,
    sample_images: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Run one forward pass and reuse its outputs in all tests.
    """
    with torch.inference_mode():
        vit_features, swin_features = dual_encoder(sample_images)

    return vit_features, swin_features


def test_base_vision_encoder_is_abstract() -> None:
    """
    BaseVisionEncoder must not be instantiated directly.
    """
    with pytest.raises(TypeError):
        BaseVisionEncoder()


def test_encoders_inherit_from_base_class(
    dual_encoder: DualVisionEncoder,
) -> None:
    """
    ViTEncoder and SwinEncoder must follow the common
    BaseVisionEncoder contract.
    """
    assert isinstance(
        dual_encoder.vit_encoder,
        BaseVisionEncoder,
    )

    assert isinstance(
        dual_encoder.swin_encoder,
        BaseVisionEncoder,
    )


def test_dual_encoder_contains_correct_encoder_types(
    dual_encoder: DualVisionEncoder,
) -> None:
    """
    DualVisionEncoder must contain one ViT encoder
    and one Swin encoder.
    """
    assert isinstance(
        dual_encoder.vit_encoder,
        ViTEncoder,
    )

    assert isinstance(
        dual_encoder.swin_encoder,
        SwinEncoder,
    )


def test_vit_encoder_is_frozen(
    dual_encoder: DualVisionEncoder,
) -> None:
    """
    All ViT parameters must be frozen when
    freeze_encoders=True.
    """
    parameters = dual_encoder.vit_encoder.parameters()

    assert all(
        parameter.requires_grad is False
        for parameter in parameters
    )


def test_swin_encoder_is_frozen(
    dual_encoder: DualVisionEncoder,
) -> None:
    """
    All Swin parameters must be frozen when
    freeze_encoders=True.
    """
    parameters = dual_encoder.swin_encoder.parameters()

    assert all(
        parameter.requires_grad is False
        for parameter in parameters
    )


def test_vit_hidden_size(
    dual_encoder: DualVisionEncoder,
) -> None:
    """
    Stored ViT hidden size must match the model configuration.
    """
    expected_hidden_size = (
        dual_encoder.vit_encoder.model.config.hidden_size
    )

    assert (
        dual_encoder.vit_encoder.hidden_size
        == expected_hidden_size
    )


def test_swin_hidden_size(
    dual_encoder: DualVisionEncoder,
) -> None:
    """
    Stored Swin hidden size must match the model configuration.
    """
    expected_hidden_size = (
        dual_encoder.swin_encoder.model.config.hidden_size
    )

    assert (
        dual_encoder.swin_encoder.hidden_size
        == expected_hidden_size
    )


def test_vit_output_shape(
    dual_encoder: DualVisionEncoder,
    encoder_outputs: tuple[Tensor, Tensor],
) -> None:
    """
    ViT output must have this general shape:

        (batch_size, number_of_tokens, hidden_size)
    """
    vit_features, _ = encoder_outputs

    assert isinstance(vit_features, Tensor)
    assert vit_features.ndim == 3

    assert vit_features.shape[0] == BATCH_SIZE

    assert vit_features.shape[1] > 0

    assert (
        vit_features.shape[2]
        == dual_encoder.vit_encoder.hidden_size
    )


def test_swin_output_shape(
    dual_encoder: DualVisionEncoder,
    encoder_outputs: tuple[Tensor, Tensor],
) -> None:
    """
    Swin output must have this general shape:

        (batch_size, number_of_tokens, hidden_size)
    """
    _, swin_features = encoder_outputs

    assert isinstance(swin_features, Tensor)
    assert swin_features.ndim == 3

    assert swin_features.shape[0] == BATCH_SIZE

    assert swin_features.shape[1] > 0

    assert (
        swin_features.shape[2]
        == dual_encoder.swin_encoder.hidden_size
    )


def test_dual_encoder_returns_two_different_outputs(
    encoder_outputs: tuple[Tensor, Tensor],
) -> None:
    """
    DualVisionEncoder must return separate outputs
    for ViT and Swin.
    """
    vit_features, swin_features = encoder_outputs

    assert vit_features is not swin_features

    assert vit_features.data_ptr() != swin_features.data_ptr()


def test_encoder_outputs_are_finite(
    encoder_outputs: tuple[Tensor, Tensor],
) -> None:
    """
    Encoder outputs must not contain NaN or infinity.
    """
    vit_features, swin_features = encoder_outputs

    assert torch.isfinite(vit_features).all()
    assert torch.isfinite(swin_features).all()