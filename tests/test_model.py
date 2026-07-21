"""
Tests for the complete proposed image-captioning model.

These tests use lightweight dummy modules instead of downloading
real ViT, Swin, or T5 checkpoints.

Run from the project root with:

    python -m pytest tests/test_model.py -v
"""

from typing import Any

import pytest
import torch
from torch import Tensor, nn
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.proposed.model import (
    BaseImageCaptioningModel,
    DualCaptionModelOutput,
    ProposedImageCaptioningModel,
)


BATCH_SIZE = 2
IMAGE_CHANNELS = 3
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

VIT_TOKENS = 197
SWIN_TOKENS = 49

VIT_ENCODER_DIM = 16
SWIN_ENCODER_DIM = 24

DECODER_HIDDEN_SIZE = 32
VOCAB_SIZE = 64
CAPTION_LENGTH = 6


class DummyDualVisionEncoder(nn.Module):
    """
    Lightweight replacement for DualVisionEncoder.

    It returns deterministic ViT and Swin feature tensors with
    different token counts and hidden dimensions.
    """

    def forward(
        self,
        images: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Produce dummy visual features from an image batch.
        """
        batch_size = images.shape[0]

        vit_features = images.new_full(
            (
                batch_size,
                VIT_TOKENS,
                VIT_ENCODER_DIM,
            ),
            fill_value=1.0,
        )

        swin_features = images.new_full(
            (
                batch_size,
                SWIN_TOKENS,
                SWIN_ENCODER_DIM,
            ),
            fill_value=2.0,
        )

        return vit_features, swin_features


class DummyVisionProjection(nn.Module):
    """
    Lightweight projection branch.

    The output_dim attribute matches the interface expected by
    ProposedImageCaptioningModel.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(
            input_dim,
            output_dim,
        )

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        """
        Project the final feature dimension.
        """
        return self.linear(features)


class DummyDualVisionProjection(nn.Module):
    """
    Lightweight replacement for DualVisionProjection.
    """

    def __init__(
        self,
        vit_output_dim: int = DECODER_HIDDEN_SIZE,
        swin_output_dim: int = DECODER_HIDDEN_SIZE,
    ) -> None:
        super().__init__()

        self.vit_projection = DummyVisionProjection(
            input_dim=VIT_ENCODER_DIM,
            output_dim=vit_output_dim,
        )

        self.swin_projection = DummyVisionProjection(
            input_dim=SWIN_ENCODER_DIM,
            output_dim=swin_output_dim,
        )

    def forward(
        self,
        vit_features: Tensor,
        swin_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Project both encoder feature tensors.
        """
        projected_vit_features = self.vit_projection(
            vit_features
        )

        projected_swin_features = self.swin_projection(
            swin_features
        )

        return (
            projected_vit_features,
            projected_swin_features,
        )


class DummyCaptionDecoder(nn.Module):
    """
    Lightweight replacement for T5CaptionDecoder.

    It records the arguments received during forward so the tests
    can verify that the complete model forwards the correct
    features, masks, labels, and decoder input IDs.
    """

    def __init__(
        self,
        hidden_size: int = DECODER_HIDDEN_SIZE,
        vocab_size: int = VOCAB_SIZE,
        branch_value: float = 1.0,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.branch_value = branch_value

        self.last_visual_features: Tensor | None = None
        self.last_visual_attention_mask: Tensor | None = None
        self.last_decoder_input_ids: Tensor | None = None
        self.last_labels: Tensor | None = None

    def forward(
        self,
        visual_features: Tensor,
        visual_attention_mask: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> Seq2SeqLMOutput:
        """
        Return deterministic dummy logits and optional loss.
        """
        if visual_features.ndim != 3:
            raise ValueError(
                "visual_features must be three-dimensional."
            )

        if visual_features.shape[-1] != self.hidden_size:
            raise ValueError(
                "visual feature hidden dimension does not match "
                "the decoder hidden size."
            )

        if labels is None and decoder_input_ids is None:
            raise ValueError(
                "Either labels or decoder_input_ids must be provided."
            )

        self.last_visual_features = visual_features
        self.last_visual_attention_mask = visual_attention_mask
        self.last_decoder_input_ids = decoder_input_ids
        self.last_labels = labels

        if labels is not None:
            sequence_length = labels.shape[1]
        else:
            assert decoder_input_ids is not None
            sequence_length = decoder_input_ids.shape[1]

        logits = visual_features.new_full(
            (
                visual_features.shape[0],
                sequence_length,
                self.vocab_size,
            ),
            fill_value=self.branch_value,
        )

        loss: Tensor | None

        if labels is not None:
            loss = visual_features.new_tensor(
                self.branch_value
            )
        else:
            loss = None

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
        )


@pytest.fixture
def encoder() -> DummyDualVisionEncoder:
    """
    Create the dummy dual encoder.
    """
    return DummyDualVisionEncoder()


@pytest.fixture
def projection() -> DummyDualVisionProjection:
    """
    Create compatible projection branches.
    """
    return DummyDualVisionProjection()


@pytest.fixture
def vit_decoder() -> DummyCaptionDecoder:
    """
    Create a decoder assigned to the ViT branch.
    """
    return DummyCaptionDecoder(
        branch_value=1.0,
    )


@pytest.fixture
def swin_decoder() -> DummyCaptionDecoder:
    """
    Create a decoder assigned to the Swin branch.
    """
    return DummyCaptionDecoder(
        branch_value=2.0,
    )


@pytest.fixture
def model(
    encoder: DummyDualVisionEncoder,
    projection: DummyDualVisionProjection,
    vit_decoder: DummyCaptionDecoder,
    swin_decoder: DummyCaptionDecoder,
) -> ProposedImageCaptioningModel:
    """
    Create the complete proposed model with automatic masks.
    """
    caption_model = ProposedImageCaptioningModel(
        encoder=encoder,
        projection=projection,
        vit_decoder=vit_decoder,
        swin_decoder=swin_decoder,
        create_visual_masks=True,
    )

    caption_model.eval()

    return caption_model


@pytest.fixture
def images() -> Tensor:
    """
    Create a valid RGB image batch.
    """
    torch.manual_seed(42)

    return torch.randn(
        BATCH_SIZE,
        IMAGE_CHANNELS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )


@pytest.fixture
def labels() -> Tensor:
    """
    Create tokenized target captions.

    Padding locations use -100.
    """
    return torch.tensor(
        [
            [5, 6, 7, 8, 1, -100],
            [9, 10, 11, 1, -100, -100],
        ],
        dtype=torch.long,
    )


@pytest.fixture
def decoder_input_ids() -> Tensor:
    """
    Create previously generated caption tokens.
    """
    return torch.tensor(
        [
            [0, 5, 6, 7],
            [0, 9, 10, 11],
        ],
        dtype=torch.long,
    )


def build_model(
    *,
    encoder: nn.Module | None = None,
    projection: nn.Module | None = None,
    vit_decoder: nn.Module | None = None,
    swin_decoder: nn.Module | None = None,
    create_visual_masks: bool = True,
) -> ProposedImageCaptioningModel:
    """
    Build a complete dummy model for individual tests.
    """
    actual_encoder = (
        encoder
        if encoder is not None
        else DummyDualVisionEncoder()
    )

    actual_projection = (
        projection
        if projection is not None
        else DummyDualVisionProjection()
    )

    actual_vit_decoder = (
        vit_decoder
        if vit_decoder is not None
        else DummyCaptionDecoder(branch_value=1.0)
    )

    actual_swin_decoder = (
        swin_decoder
        if swin_decoder is not None
        else DummyCaptionDecoder(branch_value=2.0)
    )

    return ProposedImageCaptioningModel(
        encoder=actual_encoder,  # type: ignore[arg-type]
        projection=actual_projection,  # type: ignore[arg-type]
        vit_decoder=actual_vit_decoder,  # type: ignore[arg-type]
        swin_decoder=actual_swin_decoder,  # type: ignore[arg-type]
        create_visual_masks=create_visual_masks,
    )


def test_base_image_captioning_model_is_abstract() -> None:
    """
    The abstract base model must not be instantiated directly.
    """
    with pytest.raises(TypeError):
        BaseImageCaptioningModel()


def test_model_stores_supplied_modules(
    model: ProposedImageCaptioningModel,
    encoder: DummyDualVisionEncoder,
    projection: DummyDualVisionProjection,
    vit_decoder: DummyCaptionDecoder,
    swin_decoder: DummyCaptionDecoder,
) -> None:
    """
    The model must store the exact injected module instances.
    """
    assert model.encoder is encoder
    assert model.projection is projection
    assert model.vit_decoder is vit_decoder
    assert model.swin_decoder is swin_decoder


def test_model_uses_separate_decoders(
    model: ProposedImageCaptioningModel,
) -> None:
    """
    ViT and Swin branches must not share the same decoder object.
    """
    assert model.vit_decoder is not model.swin_decoder


@pytest.mark.parametrize(
    ("argument_name", "invalid_value"),
    [
        ("encoder", "not-an-encoder"),
        ("projection", 123),
        ("vit_decoder", object()),
        ("swin_decoder", None),
    ],
)
def test_model_rejects_non_module_dependencies(
    argument_name: str,
    invalid_value: Any,
) -> None:
    """
    Every injected dependency must inherit from nn.Module.
    """
    arguments: dict[str, Any] = {
        "encoder": DummyDualVisionEncoder(),
        "projection": DummyDualVisionProjection(),
        "vit_decoder": DummyCaptionDecoder(
            branch_value=1.0
        ),
        "swin_decoder": DummyCaptionDecoder(
            branch_value=2.0
        ),
    }

    arguments[argument_name] = invalid_value

    with pytest.raises(TypeError):
        ProposedImageCaptioningModel(
            encoder=arguments["encoder"],
            projection=arguments["projection"],
            vit_decoder=arguments["vit_decoder"],
            swin_decoder=arguments["swin_decoder"],
        )


def test_model_rejects_vit_projection_dimension_mismatch() -> None:
    """
    ViT projection output dimension must match its decoder.
    """
    incompatible_projection = DummyDualVisionProjection(
        vit_output_dim=DECODER_HIDDEN_SIZE + 1,
        swin_output_dim=DECODER_HIDDEN_SIZE,
    )

    with pytest.raises(ValueError):
        build_model(
            projection=incompatible_projection,
        )


def test_model_rejects_swin_projection_dimension_mismatch() -> None:
    """
    Swin projection output dimension must match its decoder.
    """
    incompatible_projection = DummyDualVisionProjection(
        vit_output_dim=DECODER_HIDDEN_SIZE,
        swin_output_dim=DECODER_HIDDEN_SIZE + 1,
    )

    with pytest.raises(ValueError):
        build_model(
            projection=incompatible_projection,
        )


def test_validate_images_accepts_valid_rgb_batch(
    images: Tensor,
) -> None:
    """
    A four-dimensional RGB Tensor must pass validation.
    """
    ProposedImageCaptioningModel._validate_images(images)


def test_validate_images_rejects_non_tensor() -> None:
    """
    Image input must be a Tensor.
    """
    with pytest.raises(TypeError):
        ProposedImageCaptioningModel._validate_images(
            [[1, 2, 3]]  # type: ignore[arg-type]
        )


def test_validate_images_rejects_wrong_number_of_dimensions() -> None:
    """
    Images must have batch, channel, height, and width dimensions.
    """
    invalid_images = torch.randn(
        IMAGE_CHANNELS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )

    with pytest.raises(ValueError):
        ProposedImageCaptioningModel._validate_images(
            invalid_images
        )


def test_validate_images_rejects_non_rgb_images() -> None:
    """
    The channel dimension must contain three RGB channels.
    """
    grayscale_images = torch.randn(
        BATCH_SIZE,
        1,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )

    with pytest.raises(ValueError):
        ProposedImageCaptioningModel._validate_images(
            grayscale_images
        )


def test_create_full_visual_attention_mask() -> None:
    """
    The generated mask must contain one value per visual token.
    """
    visual_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        DECODER_HIDDEN_SIZE,
    )

    mask = (
        ProposedImageCaptioningModel
        ._create_full_visual_attention_mask(
            visual_features
        )
    )

    assert mask.shape == (
        BATCH_SIZE,
        VIT_TOKENS,
    )

    assert mask.dtype == torch.long
    assert mask.device == visual_features.device
    assert torch.all(mask == 1)


def test_validate_visual_mask_accepts_valid_mask() -> None:
    """
    A correctly shaped mask must pass validation.
    """
    visual_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        DECODER_HIDDEN_SIZE,
    )

    mask = torch.ones(
        BATCH_SIZE,
        VIT_TOKENS,
        dtype=torch.long,
    )

    ProposedImageCaptioningModel._validate_visual_attention_mask(
        mask=mask,
        visual_features=visual_features,
        mask_name="vit_visual_attention_mask",
    )


def test_validate_visual_mask_rejects_non_two_dimensional_mask() -> None:
    """
    Visual masks must have shape (batch_size, visual_tokens).
    """
    visual_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        DECODER_HIDDEN_SIZE,
    )

    invalid_mask = torch.ones(
        BATCH_SIZE,
        VIT_TOKENS,
        1,
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        ProposedImageCaptioningModel._validate_visual_attention_mask(
            mask=invalid_mask,
            visual_features=visual_features,
            mask_name="vit_visual_attention_mask",
        )


def test_validate_visual_mask_rejects_wrong_shape() -> None:
    """
    Mask shape must match the first two feature dimensions.
    """
    visual_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        DECODER_HIDDEN_SIZE,
    )

    invalid_mask = torch.ones(
        BATCH_SIZE,
        VIT_TOKENS + 1,
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        ProposedImageCaptioningModel._validate_visual_attention_mask(
            mask=invalid_mask,
            visual_features=visual_features,
            mask_name="vit_visual_attention_mask",
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the device mismatch test.",
)
def test_validate_visual_mask_rejects_device_mismatch() -> None:
    """
    Mask and visual features must be placed on the same device.
    """
    visual_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        DECODER_HIDDEN_SIZE,
        device="cuda",
    )

    cpu_mask = torch.ones(
        BATCH_SIZE,
        VIT_TOKENS,
        dtype=torch.long,
        device="cpu",
    )

    with pytest.raises(ValueError):
        ProposedImageCaptioningModel._validate_visual_attention_mask(
            mask=cpu_mask,
            visual_features=visual_features,
            mask_name="vit_visual_attention_mask",
        )


def test_encode_images_returns_projected_features(
    model: ProposedImageCaptioningModel,
    images: Tensor,
) -> None:
    """
    encode_images must run both encoders and projections.
    """
    projected_vit, projected_swin = model.encode_images(
        images
    )

    assert projected_vit.shape == (
        BATCH_SIZE,
        VIT_TOKENS,
        DECODER_HIDDEN_SIZE,
    )

    assert projected_swin.shape == (
        BATCH_SIZE,
        SWIN_TOKENS,
        DECODER_HIDDEN_SIZE,
    )


def test_encode_images_preserves_device(
    model: ProposedImageCaptioningModel,
    images: Tensor,
) -> None:
    """
    Projected features must remain on the image device.
    """
    projected_vit, projected_swin = model.encode_images(
        images
    )

    assert projected_vit.device == images.device
    assert projected_swin.device == images.device


def test_training_forward_returns_dual_output(
    model: ProposedImageCaptioningModel,
    images: Tensor,
    labels: Tensor,
) -> None:
    """
    Training forward must return loss and logits for both branches.
    """
    outputs = model(
        images=images,
        labels=labels,
    )

    assert isinstance(
        outputs,
        DualCaptionModelOutput,
    )

    assert isinstance(
        outputs.vit_outputs,
        Seq2SeqLMOutput,
    )

    assert isinstance(
        outputs.swin_outputs,
        Seq2SeqLMOutput,
    )

    assert outputs.vit_outputs.loss is not None
    assert outputs.swin_outputs.loss is not None

    assert outputs.vit_outputs.loss.ndim == 0
    assert outputs.swin_outputs.loss.ndim == 0

    assert outputs.vit_outputs.logits.shape == (
        BATCH_SIZE,
        CAPTION_LENGTH,
        VOCAB_SIZE,
    )

    assert outputs.swin_outputs.logits.shape == (
        BATCH_SIZE,
        CAPTION_LENGTH,
        VOCAB_SIZE,
    )


def test_training_forward_creates_automatic_visual_masks(
    model: ProposedImageCaptioningModel,
    images: Tensor,
    labels: Tensor,
    vit_decoder: DummyCaptionDecoder,
    swin_decoder: DummyCaptionDecoder,
) -> None:
    """
    Full one-valued masks must be created automatically.
    """
    model(
        images=images,
        labels=labels,
    )

    vit_mask = vit_decoder.last_visual_attention_mask
    swin_mask = swin_decoder.last_visual_attention_mask

    assert vit_mask is not None
    assert swin_mask is not None

    assert vit_mask.shape == (
        BATCH_SIZE,
        VIT_TOKENS,
    )

    assert swin_mask.shape == (
        BATCH_SIZE,
        SWIN_TOKENS,
    )

    assert torch.all(vit_mask == 1)
    assert torch.all(swin_mask == 1)


def test_training_forward_sends_labels_to_both_decoders(
    model: ProposedImageCaptioningModel,
    images: Tensor,
    labels: Tensor,
    vit_decoder: DummyCaptionDecoder,
    swin_decoder: DummyCaptionDecoder,
) -> None:
    """
    The same target captions must reach both branches.
    """
    model(
        images=images,
        labels=labels,
    )

    assert vit_decoder.last_labels is labels
    assert swin_decoder.last_labels is labels

    assert vit_decoder.last_decoder_input_ids is None
    assert swin_decoder.last_decoder_input_ids is None


def test_inference_forward_uses_decoder_input_ids(
    model: ProposedImageCaptioningModel,
    images: Tensor,
    decoder_input_ids: Tensor,
    vit_decoder: DummyCaptionDecoder,
    swin_decoder: DummyCaptionDecoder,
) -> None:
    """
    Inference must forward previously generated tokens.
    """
    outputs = model(
        images=images,
        decoder_input_ids=decoder_input_ids,
    )

    assert outputs.vit_outputs.loss is None
    assert outputs.swin_outputs.loss is None

    assert outputs.vit_outputs.logits.shape == (
        BATCH_SIZE,
        decoder_input_ids.shape[1],
        VOCAB_SIZE,
    )

    assert outputs.swin_outputs.logits.shape == (
        BATCH_SIZE,
        decoder_input_ids.shape[1],
        VOCAB_SIZE,
    )

    assert (
        vit_decoder.last_decoder_input_ids
        is decoder_input_ids
    )

    assert (
        swin_decoder.last_decoder_input_ids
        is decoder_input_ids
    )


def test_forward_passes_custom_masks_to_decoders(
    model: ProposedImageCaptioningModel,
    images: Tensor,
    labels: Tensor,
    vit_decoder: DummyCaptionDecoder,
    swin_decoder: DummyCaptionDecoder,
) -> None:
    """
    User-provided masks must be validated and forwarded unchanged.
    """
    vit_mask = torch.ones(
        BATCH_SIZE,
        VIT_TOKENS,
        dtype=torch.long,
    )

    vit_mask[:, -2:] = 0

    swin_mask = torch.ones(
        BATCH_SIZE,
        SWIN_TOKENS,
        dtype=torch.long,
    )

    swin_mask[:, -1] = 0

    model(
        images=images,
        labels=labels,
        vit_visual_attention_mask=vit_mask,
        swin_visual_attention_mask=swin_mask,
    )

    assert (
        vit_decoder.last_visual_attention_mask
        is vit_mask
    )

    assert (
        swin_decoder.last_visual_attention_mask
        is swin_mask
    )


def test_forward_rejects_wrong_vit_mask_shape(
    model: ProposedImageCaptioningModel,
    images: Tensor,
    labels: Tensor,
) -> None:
    """
    Invalid ViT mask shape must be rejected.
    """
    invalid_vit_mask = torch.ones(
        BATCH_SIZE,
        VIT_TOKENS + 1,
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        model(
            images=images,
            labels=labels,
            vit_visual_attention_mask=invalid_vit_mask,
        )


def test_forward_rejects_wrong_swin_mask_shape(
    model: ProposedImageCaptioningModel,
    images: Tensor,
    labels: Tensor,
) -> None:
    """
    Invalid Swin mask shape must be rejected.
    """
    invalid_swin_mask = torch.ones(
        BATCH_SIZE,
        SWIN_TOKENS + 1,
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        model(
            images=images,
            labels=labels,
            swin_visual_attention_mask=invalid_swin_mask,
        )


def test_forward_can_disable_automatic_masks(
    images: Tensor,
    labels: Tensor,
) -> None:
    """
    When create_visual_masks=False, None must reach both decoders.
    """
    vit_decoder = DummyCaptionDecoder(
        branch_value=1.0
    )

    swin_decoder = DummyCaptionDecoder(
        branch_value=2.0
    )

    model = ProposedImageCaptioningModel(
        encoder=DummyDualVisionEncoder(),
        projection=DummyDualVisionProjection(),
        vit_decoder=vit_decoder,
        swin_decoder=swin_decoder,
        create_visual_masks=False,
    )

    model(
        images=images,
        labels=labels,
    )

    assert vit_decoder.last_visual_attention_mask is None
    assert swin_decoder.last_visual_attention_mask is None


def test_two_branches_produce_independent_outputs(
    model: ProposedImageCaptioningModel,
    images: Tensor,
    labels: Tensor,
) -> None:
    """
    The ViT and Swin decoders must produce separate outputs.
    """
    outputs = model(
        images=images,
        labels=labels,
    )

    expected_vit_logits = torch.ones_like(
        outputs.vit_outputs.logits
    )

    expected_swin_logits = torch.full_like(
        outputs.swin_outputs.logits,
        fill_value=2.0,
    )

    assert torch.equal(
        outputs.vit_outputs.logits,
        expected_vit_logits,
    )

    assert torch.equal(
        outputs.swin_outputs.logits,
        expected_swin_logits,
    )

    assert not torch.equal(
        outputs.vit_outputs.logits,
        outputs.swin_outputs.logits,
    )


def test_forward_sends_correct_visual_token_counts(
    model: ProposedImageCaptioningModel,
    images: Tensor,
    labels: Tensor,
    vit_decoder: DummyCaptionDecoder,
    swin_decoder: DummyCaptionDecoder,
) -> None:
    """
    Each decoder must receive features from its own encoder branch.
    """
    model(
        images=images,
        labels=labels,
    )

    assert vit_decoder.last_visual_features is not None
    assert swin_decoder.last_visual_features is not None

    assert vit_decoder.last_visual_features.shape[1] == VIT_TOKENS
    assert (
        swin_decoder.last_visual_features.shape[1]
        == SWIN_TOKENS
    )

    assert (
        vit_decoder.last_visual_features.shape[-1]
        == DECODER_HIDDEN_SIZE
    )

    assert (
        swin_decoder.last_visual_features.shape[-1]
        == DECODER_HIDDEN_SIZE
    )