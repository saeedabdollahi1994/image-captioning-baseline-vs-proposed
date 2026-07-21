"""
Tests for the T5 caption decoder.

The tests use a tiny randomly initialized T5 model instead of
downloading the real t5-small checkpoint.

Run from the project root with:

    python -m pytest tests/test_t5_decoder.py -v
"""

import pytest
import torch
from torch import Tensor
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.proposed.t5_decoder import (
    BaseCaptionDecoder,
    T5CaptionDecoder,
)


BATCH_SIZE = 2
VISUAL_TOKENS = 5
CAPTION_LENGTH = 6

HIDDEN_SIZE = 32
VOCAB_SIZE = 64

PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
DECODER_START_TOKEN_ID = 0


def build_tiny_t5_config() -> T5Config:
    """
    Create a very small T5 configuration for fast unit tests.

    Returns:
        A T5Config with small hidden dimensions and one layer.
    """
    return T5Config(
        vocab_size=VOCAB_SIZE,
        d_model=HIDDEN_SIZE,
        d_kv=8,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        dropout_rate=0.0,
        pad_token_id=PAD_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        decoder_start_token_id=DECODER_START_TOKEN_ID,
    )


@pytest.fixture(autouse=True)
def replace_pretrained_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Replace from_pretrained with a tiny local T5 model.

    This prevents network access and makes the tests faster.
    """

    def fake_from_pretrained(
        model_name: str,
        *args,
        **kwargs,
    ) -> T5ForConditionalGeneration:
        del model_name, args, kwargs

        config = build_tiny_t5_config()

        return T5ForConditionalGeneration(config)

    monkeypatch.setattr(
        T5ForConditionalGeneration,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )


@pytest.fixture
def decoder() -> T5CaptionDecoder:
    """
    Create a trainable T5 caption decoder.
    """
    model = T5CaptionDecoder(
        model_name="tiny-t5",
        freeze_decoder=False,
    )

    model.eval()

    return model


@pytest.fixture
def frozen_decoder() -> T5CaptionDecoder:
    """
    Create a frozen T5 caption decoder.
    """
    model = T5CaptionDecoder(
        model_name="tiny-t5",
        freeze_decoder=True,
    )

    model.eval()

    return model


@pytest.fixture
def visual_features() -> Tensor:
    """
    Create projected visual features.

    Shape:
        (batch_size, visual_tokens, hidden_size)
    """
    torch.manual_seed(42)

    return torch.randn(
        BATCH_SIZE,
        VISUAL_TOKENS,
        HIDDEN_SIZE,
    )


@pytest.fixture
def visual_attention_mask() -> Tensor:
    """
    Create a visual attention mask.

    The final two visual tokens of the second sample
    are considered invalid.
    """
    return torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )


@pytest.fixture
def labels() -> Tensor:
    """
    Create target caption token IDs.

    Padding positions use -100 so they are ignored
    during loss calculation.
    """
    return torch.tensor(
        [
            [5, 6, 7, 8, EOS_TOKEN_ID, -100],
            [9, 10, 11, EOS_TOKEN_ID, -100, -100],
        ],
        dtype=torch.long,
    )


@pytest.fixture
def decoder_input_ids() -> Tensor:
    """
    Create caption tokens already given to the decoder.

    Shape:
        (batch_size, caption_length)
    """
    return torch.tensor(
        [
            [PAD_TOKEN_ID, 5, 6, 7],
            [PAD_TOKEN_ID, 9, 10, 11],
        ],
        dtype=torch.long,
    )


def test_base_caption_decoder_is_abstract() -> None:
    """
    BaseCaptionDecoder must not be instantiated directly.
    """
    with pytest.raises(TypeError):
        BaseCaptionDecoder()


def test_t5_decoder_inherits_from_base_class(
    decoder: T5CaptionDecoder,
) -> None:
    """
    T5CaptionDecoder must follow the base decoder contract.
    """
    assert isinstance(
        decoder,
        BaseCaptionDecoder,
    )


def test_t5_model_is_loaded(
    decoder: T5CaptionDecoder,
) -> None:
    """
    The decoder must contain a T5 conditional-generation model.
    """
    assert isinstance(
        decoder.model,
        T5ForConditionalGeneration,
    )

    assert decoder.model_name == "tiny-t5"


def test_decoder_reads_model_configuration(
    decoder: T5CaptionDecoder,
) -> None:
    """
    hidden_size and vocab_size must match the T5 configuration.
    """
    assert decoder.hidden_size == HIDDEN_SIZE
    assert decoder.vocab_size == VOCAB_SIZE

    assert (
        decoder.hidden_size
        == decoder.model.config.d_model
    )

    assert (
        decoder.vocab_size
        == decoder.model.config.vocab_size
    )


def test_decoder_parameters_are_trainable(
    decoder: T5CaptionDecoder,
) -> None:
    """
    T5 parameters must remain trainable when
    freeze_decoder=False.
    """
    parameters = list(decoder.model.parameters())

    assert len(parameters) > 0

    assert all(
        parameter.requires_grad
        for parameter in parameters
    )


def test_frozen_decoder_parameters_do_not_require_gradients(
    frozen_decoder: T5CaptionDecoder,
) -> None:
    """
    All T5 parameters must be frozen when
    freeze_decoder=True.
    """
    parameters = list(frozen_decoder.model.parameters())

    assert len(parameters) > 0

    assert all(
        parameter.requires_grad is False
        for parameter in parameters
    )


def test_decoder_rejects_non_three_dimensional_features(
    decoder: T5CaptionDecoder,
    labels: Tensor,
) -> None:
    """
    Visual features must be a three-dimensional tensor.
    """
    invalid_features = torch.randn(
        BATCH_SIZE,
        HIDDEN_SIZE,
    )

    with pytest.raises(ValueError):
        decoder(
            visual_features=invalid_features,
            labels=labels,
        )


def test_decoder_rejects_wrong_visual_hidden_dimension(
    decoder: T5CaptionDecoder,
    labels: Tensor,
) -> None:
    """
    The final visual dimension must equal the T5 hidden size.
    """
    invalid_features = torch.randn(
        BATCH_SIZE,
        VISUAL_TOKENS,
        HIDDEN_SIZE + 1,
    )

    with pytest.raises(ValueError):
        decoder(
            visual_features=invalid_features,
            labels=labels,
        )


def test_decoder_rejects_non_two_dimensional_visual_mask(
    decoder: T5CaptionDecoder,
    visual_features: Tensor,
    labels: Tensor,
) -> None:
    """
    Visual attention mask must be two-dimensional.
    """
    invalid_mask = torch.ones(
        BATCH_SIZE,
        VISUAL_TOKENS,
        1,
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        decoder(
            visual_features=visual_features,
            visual_attention_mask=invalid_mask,
            labels=labels,
        )


def test_decoder_rejects_wrong_visual_mask_shape(
    decoder: T5CaptionDecoder,
    visual_features: Tensor,
    labels: Tensor,
) -> None:
    """
    Visual mask shape must match batch size and visual tokens.
    """
    invalid_mask = torch.ones(
        BATCH_SIZE,
        VISUAL_TOKENS + 1,
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        decoder(
            visual_features=visual_features,
            visual_attention_mask=invalid_mask,
            labels=labels,
        )


def test_decoder_rejects_non_two_dimensional_decoder_ids(
    decoder: T5CaptionDecoder,
    visual_features: Tensor,
) -> None:
    """
    decoder_input_ids must be two-dimensional.
    """
    invalid_decoder_ids = torch.ones(
        BATCH_SIZE,
        2,
        3,
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        decoder(
            visual_features=visual_features,
            decoder_input_ids=invalid_decoder_ids,
        )


def test_decoder_rejects_decoder_ids_batch_mismatch(
    decoder: T5CaptionDecoder,
    visual_features: Tensor,
) -> None:
    """
    decoder_input_ids batch size must match visual features.
    """
    invalid_decoder_ids = torch.ones(
        BATCH_SIZE + 1,
        CAPTION_LENGTH,
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        decoder(
            visual_features=visual_features,
            decoder_input_ids=invalid_decoder_ids,
        )


def test_decoder_rejects_non_two_dimensional_labels(
    decoder: T5CaptionDecoder,
    visual_features: Tensor,
) -> None:
    """
    Labels must be a two-dimensional tensor.
    """
    invalid_labels = torch.ones(
        BATCH_SIZE,
        CAPTION_LENGTH,
        1,
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        decoder(
            visual_features=visual_features,
            labels=invalid_labels,
        )


def test_decoder_rejects_labels_batch_mismatch(
    decoder: T5CaptionDecoder,
    visual_features: Tensor,
) -> None:
    """
    Labels batch size must match visual features.
    """
    invalid_labels = torch.ones(
        BATCH_SIZE + 1,
        CAPTION_LENGTH,
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        decoder(
            visual_features=visual_features,
            labels=invalid_labels,
        )


def test_decoder_requires_labels_or_decoder_input_ids(
    decoder: T5CaptionDecoder,
    visual_features: Tensor,
) -> None:
    """
    Training needs labels and inference needs decoder_input_ids.
    At least one of them must be provided.
    """
    with pytest.raises(ValueError):
        decoder(
            visual_features=visual_features,
        )


def test_training_forward_returns_loss_and_logits(
    decoder: T5CaptionDecoder,
    visual_features: Tensor,
    visual_attention_mask: Tensor,
    labels: Tensor,
) -> None:
    """
    Forward pass with labels must return training loss and logits.
    """
    with torch.inference_mode():
        outputs = decoder(
            visual_features=visual_features,
            visual_attention_mask=visual_attention_mask,
            labels=labels,
        )

    assert isinstance(
        outputs,
        Seq2SeqLMOutput,
    )

    assert outputs.loss is not None
    assert outputs.loss.ndim == 0
    assert torch.isfinite(outputs.loss)

    assert outputs.logits.shape == (
        BATCH_SIZE,
        CAPTION_LENGTH,
        VOCAB_SIZE,
    )

    assert torch.isfinite(outputs.logits).all()


def test_inference_forward_returns_logits_without_loss(
    decoder: T5CaptionDecoder,
    visual_features: Tensor,
    visual_attention_mask: Tensor,
    decoder_input_ids: Tensor,
) -> None:
    """
    Forward pass with decoder_input_ids must return logits
    without calculating training loss.
    """
    with torch.inference_mode():
        outputs = decoder(
            visual_features=visual_features,
            visual_attention_mask=visual_attention_mask,
            decoder_input_ids=decoder_input_ids,
        )

    assert isinstance(
        outputs,
        Seq2SeqLMOutput,
    )

    assert outputs.loss is None

    assert outputs.logits.shape == (
        BATCH_SIZE,
        decoder_input_ids.shape[1],
        VOCAB_SIZE,
    )

    assert torch.isfinite(outputs.logits).all()


def test_decoder_works_without_visual_attention_mask(
    decoder: T5CaptionDecoder,
    visual_features: Tensor,
    labels: Tensor,
) -> None:
    """
    Visual attention mask is optional when every visual token
    is valid.
    """
    with torch.inference_mode():
        outputs = decoder(
            visual_features=visual_features,
            visual_attention_mask=None,
            labels=labels,
        )

    assert outputs.loss is not None

    assert outputs.logits.shape == (
        BATCH_SIZE,
        CAPTION_LENGTH,
        VOCAB_SIZE,
    )


def test_masked_and_unmasked_forward_passes_are_valid(
    decoder: T5CaptionDecoder,
    visual_features: Tensor,
    visual_attention_mask: Tensor,
    labels: Tensor,
) -> None:
    """
    Both masked and unmasked decoder calls must produce
    valid finite outputs.
    """
    with torch.inference_mode():
        masked_outputs = decoder(
            visual_features=visual_features,
            visual_attention_mask=visual_attention_mask,
            labels=labels,
        )

        unmasked_outputs = decoder(
            visual_features=visual_features,
            visual_attention_mask=None,
            labels=labels,
        )

    assert torch.isfinite(masked_outputs.logits).all()
    assert torch.isfinite(unmasked_outputs.logits).all()

    assert masked_outputs.logits.shape == (
        BATCH_SIZE,
        CAPTION_LENGTH,
        VOCAB_SIZE,
    )

    assert unmasked_outputs.logits.shape == (
        BATCH_SIZE,
        CAPTION_LENGTH,
        VOCAB_SIZE,
    )