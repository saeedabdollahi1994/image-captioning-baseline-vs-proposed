"""
Tests for visual projection layers.

Run from the project root with:

    python -m pytest tests/test_projection.py -v
"""

import pytest
import torch
from torch import Tensor, nn

from src.proposed.projection import (
    BaseVisionProjection,
    DualVisionProjection,
    VisionProjection,
)


BATCH_SIZE = 2

VIT_TOKENS = 5
SWIN_TOKENS = 7

VIT_INPUT_DIM = 16
SWIN_INPUT_DIM = 24
DECODER_HIDDEN_DIM = 8

DROPOUT_RATE = 0.1


@pytest.fixture
def vit_features() -> Tensor:
    """
    Create sample ViT features.

    Shape:
        (batch_size, vit_tokens, vit_input_dim)
    """
    torch.manual_seed(42)

    return torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        VIT_INPUT_DIM,
    )


@pytest.fixture
def swin_features() -> Tensor:
    """
    Create sample Swin features.

    Shape:
        (batch_size, swin_tokens, swin_input_dim)
    """
    torch.manual_seed(43)

    return torch.randn(
        BATCH_SIZE,
        SWIN_TOKENS,
        SWIN_INPUT_DIM,
    )


@pytest.fixture
def vit_projection() -> VisionProjection:
    """
    Create a projection layer for ViT features.
    """
    projection = VisionProjection(
        input_dim=VIT_INPUT_DIM,
        output_dim=DECODER_HIDDEN_DIM,
        dropout=DROPOUT_RATE,
    )

    projection.eval()

    return projection


@pytest.fixture
def dual_projection() -> DualVisionProjection:
    """
    Create projection branches for ViT and Swin.
    """
    projection = DualVisionProjection(
        vit_input_dim=VIT_INPUT_DIM,
        swin_input_dim=SWIN_INPUT_DIM,
        decoder_hidden_dim=DECODER_HIDDEN_DIM,
        dropout=DROPOUT_RATE,
    )

    projection.eval()

    return projection


def test_base_projection_is_abstract() -> None:
    """
    BaseVisionProjection must not be instantiated directly.
    """
    with pytest.raises(TypeError):
        BaseVisionProjection()


def test_vision_projection_inherits_from_base_class(
    vit_projection: VisionProjection,
) -> None:
    """
    VisionProjection must follow the base projection contract.
    """
    assert isinstance(
        vit_projection,
        BaseVisionProjection,
    )


def test_vision_projection_contains_expected_layers(
    vit_projection: VisionProjection,
) -> None:
    """
    VisionProjection must contain Linear, LayerNorm,
    and Dropout layers.
    """
    assert isinstance(
        vit_projection.linear,
        nn.Linear,
    )

    assert isinstance(
        vit_projection.normalization,
        nn.LayerNorm,
    )

    assert isinstance(
        vit_projection.dropout,
        nn.Dropout,
    )


def test_linear_layer_dimensions(
    vit_projection: VisionProjection,
) -> None:
    """
    Linear layer dimensions must match input_dim and output_dim.
    """
    assert (
        vit_projection.linear.in_features
        == VIT_INPUT_DIM
    )

    assert (
        vit_projection.linear.out_features
        == DECODER_HIDDEN_DIM
    )


def test_layer_norm_dimension(
    vit_projection: VisionProjection,
) -> None:
    """
    LayerNorm must normalize the projected hidden dimension.
    """
    assert (
        vit_projection.normalization.normalized_shape
        == (DECODER_HIDDEN_DIM,)
    )


def test_dropout_probability(
    vit_projection: VisionProjection,
) -> None:
    """
    Dropout probability must match the configured value.
    """
    assert (
        vit_projection.dropout.p
        == pytest.approx(DROPOUT_RATE)
    )


def test_vision_projection_output_shape(
    vit_projection: VisionProjection,
    vit_features: Tensor,
) -> None:
    """
    Projection must change only the last dimension.
    """
    with torch.inference_mode():
        projected_features = vit_projection(vit_features)

    expected_shape = (
        BATCH_SIZE,
        VIT_TOKENS,
        DECODER_HIDDEN_DIM,
    )

    assert projected_features.shape == expected_shape


def test_projection_preserves_batch_and_token_dimensions(
    vit_projection: VisionProjection,
    vit_features: Tensor,
) -> None:
    """
    Projection must preserve batch size and token count.
    """
    with torch.inference_mode():
        projected_features = vit_projection(vit_features)

    assert (
        projected_features.shape[0]
        == vit_features.shape[0]
    )

    assert (
        projected_features.shape[1]
        == vit_features.shape[1]
    )


def test_projection_rejects_non_three_dimensional_input(
    vit_projection: VisionProjection,
) -> None:
    """
    Projection must reject tensors that are not three-dimensional.
    """
    invalid_features = torch.randn(
        BATCH_SIZE,
        VIT_INPUT_DIM,
    )

    with pytest.raises(ValueError):
        vit_projection(invalid_features)


def test_projection_rejects_wrong_input_dimension(
    vit_projection: VisionProjection,
) -> None:
    """
    Projection must reject an incorrect final feature dimension.
    """
    wrong_input_dim = VIT_INPUT_DIM + 1

    invalid_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        wrong_input_dim,
    )

    with pytest.raises(ValueError):
        vit_projection(invalid_features)


def test_projection_output_is_finite(
    vit_projection: VisionProjection,
    vit_features: Tensor,
) -> None:
    """
    Projection output must not contain NaN or infinity.
    """
    with torch.inference_mode():
        projected_features = vit_projection(vit_features)

    assert torch.isfinite(projected_features).all()


def test_projection_is_deterministic_in_eval_mode(
    vit_projection: VisionProjection,
    vit_features: Tensor,
) -> None:
    """
    Dropout must be disabled in evaluation mode, so repeated
    forward passes must produce the same result.
    """
    with torch.inference_mode():
        first_output = vit_projection(vit_features)
        second_output = vit_projection(vit_features)

    assert torch.allclose(
        first_output,
        second_output,
    )


def test_dual_projection_contains_two_projection_branches(
    dual_projection: DualVisionProjection,
) -> None:
    """
    DualVisionProjection must contain separate projection
    branches for ViT and Swin.
    """
    assert isinstance(
        dual_projection.vit_projection,
        VisionProjection,
    )

    assert isinstance(
        dual_projection.swin_projection,
        VisionProjection,
    )

    assert (
        dual_projection.vit_projection
        is not dual_projection.swin_projection
    )


def test_dual_projection_uses_correct_input_dimensions(
    dual_projection: DualVisionProjection,
) -> None:
    """
    ViT and Swin branches must use their own input dimensions.
    """
    assert (
        dual_projection.vit_projection.input_dim
        == VIT_INPUT_DIM
    )

    assert (
        dual_projection.swin_projection.input_dim
        == SWIN_INPUT_DIM
    )

    assert (
        dual_projection.vit_projection.output_dim
        == DECODER_HIDDEN_DIM
    )

    assert (
        dual_projection.swin_projection.output_dim
        == DECODER_HIDDEN_DIM
    )


def test_dual_projection_output_shapes(
    dual_projection: DualVisionProjection,
    vit_features: Tensor,
    swin_features: Tensor,
) -> None:
    """
    Both branches must project their inputs to the decoder
    hidden dimension while preserving their token counts.
    """
    with torch.inference_mode():
        projected_vit, projected_swin = dual_projection(
            vit_features,
            swin_features,
        )

    assert projected_vit.shape == (
        BATCH_SIZE,
        VIT_TOKENS,
        DECODER_HIDDEN_DIM,
    )

    assert projected_swin.shape == (
        BATCH_SIZE,
        SWIN_TOKENS,
        DECODER_HIDDEN_DIM,
    )


def test_dual_projection_outputs_are_separate(
    dual_projection: DualVisionProjection,
    vit_features: Tensor,
    swin_features: Tensor,
) -> None:
    """
    ViT and Swin projection branches must return separate tensors.
    """
    with torch.inference_mode():
        projected_vit, projected_swin = dual_projection(
            vit_features,
            swin_features,
        )

    assert projected_vit is not projected_swin

    assert (
        projected_vit.data_ptr()
        != projected_swin.data_ptr()
    )


def test_projection_parameters_require_gradients(
    vit_projection: VisionProjection,
) -> None:
    """
    Projection layers are trainable and their parameters
    must require gradients.
    """
    parameters = list(vit_projection.parameters())

    assert len(parameters) > 0

    assert all(
        parameter.requires_grad
        for parameter in parameters
    )