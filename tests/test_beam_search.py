"""
Tests for dual-branch beam-search caption generation.

The tests use lightweight dummy models and tokenizers. No ViT,
Swin, T5, or tokenizer checkpoint is downloaded.

Run from the project root with:

    python -m pytest tests/test_beam_search.py -v
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutput

from src.proposed.beam_search import (
    BaseBeamSearchGenerator,
    BeamCandidate,
    BeamSearchConfig,
    BranchBeamSearchOutput,
    DualBeamSearchOutput,
    DualBranchBeamSearchGenerator,
)
from src.proposed.model import ProposedImageCaptioningModel
from src.proposed.t5_decoder import T5CaptionDecoder


BATCH_SIZE = 2
IMAGE_CHANNELS = 3
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32

VIT_TOKENS = 5
SWIN_TOKENS = 3

HIDDEN_SIZE = 16
VOCAB_SIZE = 64

NUM_BEAMS = 3
NUM_RETURN_SEQUENCES = 3


class DummyDualVisionEncoder(nn.Module):
    """
    Return deterministic ViT and Swin visual features.
    """

    def __init__(self) -> None:
        super().__init__()

        self.call_count = 0

    def forward(
        self,
        images: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Produce visual features for both branches.
        """
        self.call_count += 1

        batch_size = images.shape[0]

        vit_features = images.new_full(
            (
                batch_size,
                VIT_TOKENS,
                HIDDEN_SIZE,
            ),
            fill_value=1.0,
        )

        swin_features = images.new_full(
            (
                batch_size,
                SWIN_TOKENS,
                HIDDEN_SIZE,
            ),
            fill_value=2.0,
        )

        return vit_features, swin_features


class DummyProjectionBranch(nn.Module):
    """
    Identity projection exposing an output_dim attribute.
    """

    def __init__(
        self,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        """
        Return the features without changing them.
        """
        return features


class DummyDualVisionProjection(nn.Module):
    """
    Lightweight replacement for DualVisionProjection.
    """

    def __init__(
        self,
        vit_output_dim: int = HIDDEN_SIZE,
        swin_output_dim: int = HIDDEN_SIZE,
    ) -> None:
        super().__init__()

        self.vit_projection = DummyProjectionBranch(
            output_dim=vit_output_dim,
        )

        self.swin_projection = DummyProjectionBranch(
            output_dim=swin_output_dim,
        )

    def forward(
        self,
        vit_features: Tensor,
        swin_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Project both visual feature tensors.
        """
        return (
            self.vit_projection(vit_features),
            self.swin_projection(swin_features),
        )


class DummyGenerationModel(nn.Module):
    """
    Lightweight object implementing a generate method.

    It creates deterministic sequences and deliberately unsorted
    sequence scores so candidate sorting can be tested.
    """

    def __init__(
        self,
        branch_token: int,
        score_offset: float = 0.0,
    ) -> None:
        super().__init__()

        self.branch_token = branch_token
        self.score_offset = score_offset

        self.last_kwargs: dict[str, Any] | None = None

        self.training_during_generate: bool | None = None
        self.grad_enabled_during_generate: bool | None = None

        self.return_sequence_scores = True
        self.candidate_count_delta = 0
        self.raise_generation_error = False

    def generate(
        self,
        **kwargs: Any,
    ) -> SimpleNamespace:
        """
        Return deterministic dummy beam-search results.
        """
        if self.raise_generation_error:
            raise RuntimeError(
                "Intentional dummy generation error."
            )

        self.last_kwargs = dict(kwargs)

        self.training_during_generate = self.training
        self.grad_enabled_during_generate = (
            torch.is_grad_enabled()
        )

        encoder_outputs = kwargs["encoder_outputs"]

        visual_features = (
            encoder_outputs.last_hidden_state
        )

        batch_size = visual_features.shape[0]
        device = visual_features.device

        num_return_sequences = int(
            kwargs["num_return_sequences"]
        )

        expected_count = (
            batch_size * num_return_sequences
        )

        candidate_count = (
            expected_count
            + self.candidate_count_delta
        )

        score_pattern = [
            0.10,
            0.90,
            0.50,
            0.70,
            0.30,
        ]

        sequences: list[list[int]] = []
        sequence_scores: list[float] = []

        for flat_index in range(candidate_count):
            image_index = (
                flat_index // num_return_sequences
            )

            candidate_index = (
                flat_index % num_return_sequences
            )

            sequences.append(
                [
                    0,
                    self.branch_token,
                    image_index + 2,
                    candidate_index + 10,
                    1,
                ]
            )

            score = (
                score_pattern[
                    candidate_index
                    % len(score_pattern)
                ]
                + self.score_offset
            )

            sequence_scores.append(score)

        sequence_tensor = torch.tensor(
            sequences,
            dtype=torch.long,
            device=device,
        )

        if self.return_sequence_scores:
            score_tensor: Tensor | None = torch.tensor(
                sequence_scores,
                dtype=torch.float32,
                device=device,
            )
        else:
            score_tensor = None

        return SimpleNamespace(
            sequences=sequence_tensor,
            sequences_scores=score_tensor,
        )


class DummyT5CaptionDecoder(T5CaptionDecoder):
    """
    T5CaptionDecoder-compatible dummy implementation.

    The real T5 constructor is bypassed to prevent downloading
    pretrained weights.
    """

    def __init__(
        self,
        branch_token: int,
        hidden_size: int = HIDDEN_SIZE,
        vocab_size: int = VOCAB_SIZE,
        score_offset: float = 0.0,
    ) -> None:
        nn.Module.__init__(self)

        self.model_name = "dummy-t5"
        self.freeze_decoder = False

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.model = DummyGenerationModel(
            branch_token=branch_token,
            score_offset=score_offset,
        )


class DummyTokenizer(PreTrainedTokenizerBase):
    """
    Minimal Hugging Face tokenizer-compatible object.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
    ) -> None:
        self._dummy_vocab_size = vocab_size

        super().__init__()

    def __len__(self) -> int:
        """
        Return the configured vocabulary size.
        """
        return self._dummy_vocab_size

    def batch_decode(
        self,
        sequences: Tensor | list[list[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        """
        Convert generated token IDs into deterministic text.
        """
        del (
            skip_special_tokens,
            clean_up_tokenization_spaces,
            kwargs,
        )

        if torch.is_tensor(sequences):
            sequence_rows = sequences.tolist()
        else:
            sequence_rows = sequences

        decoded_texts: list[str] = []

        for row in sequence_rows:
            decoded_texts.append(
                "  "
                f"caption-{row[1]}-{row[2]}-{row[3]}"
                "  "
            )

        return decoded_texts


def build_config() -> BeamSearchConfig:
    """
    Create a small valid beam-search configuration.
    """
    return BeamSearchConfig(
        num_beams=NUM_BEAMS,
        num_return_sequences=NUM_RETURN_SEQUENCES,
        max_new_tokens=8,
        min_new_tokens=1,
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=2,
        use_cache=True,
    )


def build_model(
    *,
    create_visual_masks: bool = True,
    vit_vocab_size: int = VOCAB_SIZE,
    swin_vocab_size: int = VOCAB_SIZE,
) -> ProposedImageCaptioningModel:
    """
    Build a complete lightweight proposed model.
    """
    return ProposedImageCaptioningModel(
        encoder=DummyDualVisionEncoder(),
        projection=DummyDualVisionProjection(),
        vit_decoder=DummyT5CaptionDecoder(
            branch_token=11,
            vocab_size=vit_vocab_size,
            score_offset=0.0,
        ),
        swin_decoder=DummyT5CaptionDecoder(
            branch_token=22,
            vocab_size=swin_vocab_size,
            score_offset=1.0,
        ),
        create_visual_masks=create_visual_masks,
    )


def build_generator(
    *,
    model: ProposedImageCaptioningModel | None = None,
    tokenizer: DummyTokenizer | None = None,
    config: BeamSearchConfig | None = None,
) -> DualBranchBeamSearchGenerator:
    """
    Build a complete beam-search generator.
    """
    resolved_model = (
        model
        if model is not None
        else build_model()
    )

    resolved_tokenizer = (
        tokenizer
        if tokenizer is not None
        else DummyTokenizer()
    )

    resolved_config = (
        config
        if config is not None
        else build_config()
    )

    return DualBranchBeamSearchGenerator(
        model=resolved_model,
        tokenizer=resolved_tokenizer,
        config=resolved_config,
    )


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
def model() -> ProposedImageCaptioningModel:
    """
    Create the lightweight proposed model.
    """
    return build_model()


@pytest.fixture
def tokenizer() -> DummyTokenizer:
    """
    Create the dummy tokenizer.
    """
    return DummyTokenizer()


@pytest.fixture
def config() -> BeamSearchConfig:
    """
    Create valid beam-search settings.
    """
    return build_config()


@pytest.fixture
def generator(
    model: ProposedImageCaptioningModel,
    tokenizer: DummyTokenizer,
    config: BeamSearchConfig,
) -> DualBranchBeamSearchGenerator:
    """
    Create the beam-search generator.
    """
    return DualBranchBeamSearchGenerator(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )


def test_base_beam_search_generator_is_abstract() -> None:
    """
    The base generator must not be instantiated directly.
    """
    with pytest.raises(TypeError):
        BaseBeamSearchGenerator()


def test_valid_beam_search_config() -> None:
    """
    A valid configuration must pass validation.
    """
    config = build_config()

    config.validate()


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "error_type"),
    [
        ("num_beams", True, TypeError),
        ("num_beams", 1, ValueError),
        ("num_return_sequences", False, TypeError),
        ("num_return_sequences", 0, ValueError),
        ("num_return_sequences", 6, ValueError),
        ("max_new_tokens", True, TypeError),
        ("max_new_tokens", 0, ValueError),
        ("min_new_tokens", False, TypeError),
        ("min_new_tokens", -1, ValueError),
        ("length_penalty", True, TypeError),
        ("length_penalty", 0.0, ValueError),
        ("length_penalty", float("inf"), ValueError),
        ("length_penalty", float("nan"), ValueError),
        ("early_stopping", 1, TypeError),
        ("no_repeat_ngram_size", True, TypeError),
        ("no_repeat_ngram_size", -1, ValueError),
        ("use_cache", "yes", TypeError),
    ],
)
def test_invalid_beam_search_config_values(
    field_name: str,
    invalid_value: Any,
    error_type: type[Exception],
) -> None:
    """
    Invalid configuration values must raise clear errors.
    """
    arguments: dict[str, Any] = {
        "num_beams": 5,
        "num_return_sequences": 5,
        "max_new_tokens": 30,
        "min_new_tokens": 1,
        "length_penalty": 1.0,
        "early_stopping": True,
        "no_repeat_ngram_size": 2,
        "use_cache": True,
    }

    arguments[field_name] = invalid_value

    config = BeamSearchConfig(**arguments)

    with pytest.raises(error_type):
        config.validate()


def test_config_rejects_min_tokens_above_maximum() -> None:
    """
    min_new_tokens cannot exceed max_new_tokens.
    """
    config = BeamSearchConfig(
        num_beams=3,
        num_return_sequences=3,
        max_new_tokens=4,
        min_new_tokens=5,
    )

    with pytest.raises(ValueError):
        config.validate()


def test_generator_stores_supplied_dependencies(
    generator: DualBranchBeamSearchGenerator,
    model: ProposedImageCaptioningModel,
    tokenizer: DummyTokenizer,
    config: BeamSearchConfig,
) -> None:
    """
    The generator must store the injected objects.
    """
    assert generator.model is model
    assert generator.tokenizer is tokenizer
    assert generator.config is config


def test_generator_creates_default_configuration(
    model: ProposedImageCaptioningModel,
    tokenizer: DummyTokenizer,
) -> None:
    """
    A default BeamSearchConfig must be created when config=None.
    """
    generator = DualBranchBeamSearchGenerator(
        model=model,
        tokenizer=tokenizer,
        config=None,
    )

    assert isinstance(
        generator.config,
        BeamSearchConfig,
    )

    assert generator.config == BeamSearchConfig()


def test_generator_rejects_wrong_model_type(
    tokenizer: DummyTokenizer,
) -> None:
    """
    The complete proposed model is required.
    """
    with pytest.raises(TypeError):
        DualBranchBeamSearchGenerator(
            model=nn.Identity(),  # type: ignore[arg-type]
            tokenizer=tokenizer,
            config=build_config(),
        )


def test_generator_rejects_wrong_tokenizer_type(
    model: ProposedImageCaptioningModel,
) -> None:
    """
    A Hugging Face tokenizer-compatible object is required.
    """
    with pytest.raises(TypeError):
        DualBranchBeamSearchGenerator(
            model=model,
            tokenizer=object(),  # type: ignore[arg-type]
            config=build_config(),
        )


def test_generator_rejects_wrong_config_type(
    model: ProposedImageCaptioningModel,
    tokenizer: DummyTokenizer,
) -> None:
    """
    config must be BeamSearchConfig or None.
    """
    with pytest.raises(TypeError):
        DualBranchBeamSearchGenerator(
            model=model,
            tokenizer=tokenizer,
            config={"num_beams": 3},  # type: ignore[arg-type]
        )


def test_generator_rejects_different_decoder_vocabularies(
    tokenizer: DummyTokenizer,
) -> None:
    """
    Both T5 branches must use the same vocabulary size.
    """
    model = build_model(
        vit_vocab_size=VOCAB_SIZE,
        swin_vocab_size=VOCAB_SIZE + 1,
    )

    with pytest.raises(ValueError):
        DualBranchBeamSearchGenerator(
            model=model,
            tokenizer=tokenizer,
            config=build_config(),
        )


def test_generator_rejects_tokenizer_larger_than_decoder(
    model: ProposedImageCaptioningModel,
) -> None:
    """
    Tokenizer IDs must fit inside the decoder vocabulary.
    """
    tokenizer = DummyTokenizer(
        vocab_size=VOCAB_SIZE + 1,
    )

    with pytest.raises(ValueError):
        DualBranchBeamSearchGenerator(
            model=model,
            tokenizer=tokenizer,
            config=build_config(),
        )


@pytest.mark.parametrize(
    "branch",
    [
        "vit",
        "swin",
    ],
)
def test_validate_branch_name_accepts_valid_names(
    branch: str,
) -> None:
    """
    ViT and Swin are valid branch names.
    """
    result = (
        DualBranchBeamSearchGenerator
        ._validate_branch_name(branch)
    )

    assert result == branch


def test_validate_branch_name_rejects_invalid_name() -> None:
    """
    Unknown branch names must be rejected.
    """
    with pytest.raises(ValueError):
        (
            DualBranchBeamSearchGenerator
            ._validate_branch_name("resnet")
        )


def test_prepare_mask_returns_supplied_mask(
    generator: DualBranchBeamSearchGenerator,
) -> None:
    """
    A valid user-provided mask must be returned unchanged.
    """
    visual_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        HIDDEN_SIZE,
    )

    mask = torch.ones(
        BATCH_SIZE,
        VIT_TOKENS,
        dtype=torch.long,
    )

    prepared_mask = (
        generator._prepare_visual_attention_mask(
            visual_features=visual_features,
            visual_attention_mask=mask,
            mask_name="vit_visual_attention_mask",
        )
    )

    assert prepared_mask is mask


def test_prepare_mask_creates_full_mask(
    generator: DualBranchBeamSearchGenerator,
) -> None:
    """
    An all-ones mask must be created when enabled.
    """
    visual_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        HIDDEN_SIZE,
    )

    mask = generator._prepare_visual_attention_mask(
        visual_features=visual_features,
        visual_attention_mask=None,
        mask_name="vit_visual_attention_mask",
    )

    assert mask is not None

    assert mask.shape == (
        BATCH_SIZE,
        VIT_TOKENS,
    )

    assert mask.dtype == torch.long
    assert mask.device == visual_features.device
    assert torch.all(mask == 1)


def test_prepare_mask_returns_none_when_disabled() -> None:
    """
    Missing masks remain None when automatic creation is disabled.
    """
    generator = build_generator(
        model=build_model(
            create_visual_masks=False,
        )
    )

    visual_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        HIDDEN_SIZE,
    )

    mask = generator._prepare_visual_attention_mask(
        visual_features=visual_features,
        visual_attention_mask=None,
        mask_name="vit_visual_attention_mask",
    )

    assert mask is None


def test_prepare_mask_rejects_invalid_shape(
    generator: DualBranchBeamSearchGenerator,
) -> None:
    """
    A supplied mask must match the feature token count.
    """
    visual_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        HIDDEN_SIZE,
    )

    invalid_mask = torch.ones(
        BATCH_SIZE,
        VIT_TOKENS + 1,
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        generator._prepare_visual_attention_mask(
            visual_features=visual_features,
            visual_attention_mask=invalid_mask,
            mask_name="vit_visual_attention_mask",
        )


def test_wrap_visual_features(
    generator: DualBranchBeamSearchGenerator,
) -> None:
    """
    Visual features must be wrapped without being copied.
    """
    visual_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        HIDDEN_SIZE,
    )

    encoder_outputs = (
        generator._wrap_visual_features(
            visual_features
        )
    )

    assert isinstance(
        encoder_outputs,
        BaseModelOutput,
    )

    assert (
        encoder_outputs.last_hidden_state
        is visual_features
    )


def test_wrap_visual_features_rejects_non_tensor(
    generator: DualBranchBeamSearchGenerator,
) -> None:
    """
    visual_features must be a Tensor.
    """
    with pytest.raises(TypeError):
        generator._wrap_visual_features(
            [[1, 2, 3]]  # type: ignore[arg-type]
        )


def test_wrap_visual_features_rejects_wrong_dimensions(
    generator: DualBranchBeamSearchGenerator,
) -> None:
    """
    Visual features must be three-dimensional.
    """
    invalid_features = torch.randn(
        BATCH_SIZE,
        HIDDEN_SIZE,
    )

    with pytest.raises(ValueError):
        generator._wrap_visual_features(
            invalid_features
        )


def test_generate_branch_returns_grouped_candidates(
    generator: DualBranchBeamSearchGenerator,
    model: ProposedImageCaptioningModel,
) -> None:
    """
    Generated candidates must be grouped by input image.
    """
    visual_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        HIDDEN_SIZE,
    )

    mask = torch.ones(
        BATCH_SIZE,
        VIT_TOKENS,
        dtype=torch.long,
    )

    output = generator._generate_branch(
        visual_features=visual_features,
        visual_attention_mask=mask,
        decoder=model.vit_decoder,
        branch="vit",
    )

    assert isinstance(
        output,
        BranchBeamSearchOutput,
    )

    assert output.branch == "vit"

    assert len(output.candidates) == BATCH_SIZE

    for image_candidates in output.candidates:
        assert (
            len(image_candidates)
            == NUM_RETURN_SEQUENCES
        )

        assert all(
            isinstance(candidate, BeamCandidate)
            for candidate in image_candidates
        )


def test_generate_branch_sorts_scores_and_rebuilds_ranks(
    generator: DualBranchBeamSearchGenerator,
    model: ProposedImageCaptioningModel,
) -> None:
    """
    Candidates must be sorted from highest to lowest beam score.
    """
    visual_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        HIDDEN_SIZE,
    )

    output = generator._generate_branch(
        visual_features=visual_features,
        visual_attention_mask=None,
        decoder=model.vit_decoder,
        branch="vit",
    )

    first_image_candidates = output.candidates[0]

    scores = [
        candidate.beam_score
        for candidate in first_image_candidates
    ]

    ranks = [
        candidate.rank
        for candidate in first_image_candidates
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    assert ranks == [1, 2, 3]

    assert first_image_candidates[0].beam_score == pytest.approx(
        0.90
    )

    assert first_image_candidates[1].beam_score == pytest.approx(
        0.50
    )

    assert first_image_candidates[2].beam_score == pytest.approx(
        0.10
    )

    # Candidate index 1 had the highest dummy score.
    assert first_image_candidates[0].token_ids[3] == 11


def test_generate_branch_creates_python_candidate_data(
    generator: DualBranchBeamSearchGenerator,
    model: ProposedImageCaptioningModel,
) -> None:
    """
    Candidate data must not retain GPU or live Tensor objects.
    """
    visual_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        HIDDEN_SIZE,
    )

    output = generator._generate_branch(
        visual_features=visual_features,
        visual_attention_mask=None,
        decoder=model.vit_decoder,
        branch="vit",
    )

    candidate = output.candidates[0][0]

    assert candidate.branch == "vit"
    assert isinstance(candidate.rank, int)
    assert isinstance(candidate.text, str)
    assert isinstance(candidate.token_ids, tuple)
    assert isinstance(candidate.beam_score, float)

    assert candidate.text == candidate.text.strip()

    assert all(
        isinstance(token_id, int)
        for token_id in candidate.token_ids
    )


def test_generate_branch_passes_generation_arguments(
    generator: DualBranchBeamSearchGenerator,
    model: ProposedImageCaptioningModel,
) -> None:
    """
    BeamSearchConfig values must reach decoder.model.generate.
    """
    visual_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        HIDDEN_SIZE,
    )

    mask = torch.ones(
        BATCH_SIZE,
        VIT_TOKENS,
        dtype=torch.long,
    )

    generator._generate_branch(
        visual_features=visual_features,
        visual_attention_mask=mask,
        decoder=model.vit_decoder,
        branch="vit",
    )

    generation_model = model.vit_decoder.model

    assert isinstance(
        generation_model,
        DummyGenerationModel,
    )

    kwargs = generation_model.last_kwargs

    assert kwargs is not None

    assert isinstance(
        kwargs["encoder_outputs"],
        BaseModelOutput,
    )

    assert kwargs["attention_mask"] is mask

    assert kwargs["num_beams"] == NUM_BEAMS

    assert (
        kwargs["num_return_sequences"]
        == NUM_RETURN_SEQUENCES
    )

    assert kwargs["max_new_tokens"] == 8
    assert kwargs["min_new_tokens"] == 1
    assert kwargs["length_penalty"] == 1.0
    assert kwargs["early_stopping"] is True
    assert kwargs["no_repeat_ngram_size"] == 2
    assert kwargs["do_sample"] is False
    assert kwargs["use_cache"] is True
    assert kwargs["return_dict_in_generate"] is True
    assert kwargs["output_scores"] is True


def test_generate_branch_rejects_wrong_decoder_type(
    generator: DualBranchBeamSearchGenerator,
) -> None:
    """
    Each branch must use a T5CaptionDecoder.
    """
    visual_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        HIDDEN_SIZE,
    )

    with pytest.raises(TypeError):
        generator._generate_branch(
            visual_features=visual_features,
            visual_attention_mask=None,
            decoder=nn.Identity(),  # type: ignore[arg-type]
            branch="vit",
        )


def test_generate_branch_rejects_hidden_dimension_mismatch(
    generator: DualBranchBeamSearchGenerator,
    model: ProposedImageCaptioningModel,
) -> None:
    """
    Projected features must match decoder.hidden_size.
    """
    invalid_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        HIDDEN_SIZE + 1,
    )

    with pytest.raises(ValueError):
        generator._generate_branch(
            visual_features=invalid_features,
            visual_attention_mask=None,
            decoder=model.vit_decoder,
            branch="vit",
        )


def test_generate_branch_requires_sequence_scores(
    generator: DualBranchBeamSearchGenerator,
    model: ProposedImageCaptioningModel,
) -> None:
    """
    Missing final beam scores must produce a runtime error.
    """
    generation_model = model.vit_decoder.model

    assert isinstance(
        generation_model,
        DummyGenerationModel,
    )

    generation_model.return_sequence_scores = False

    visual_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        HIDDEN_SIZE,
    )

    with pytest.raises(RuntimeError):
        generator._generate_branch(
            visual_features=visual_features,
            visual_attention_mask=None,
            decoder=model.vit_decoder,
            branch="vit",
        )


def test_generate_branch_rejects_wrong_candidate_count(
    generator: DualBranchBeamSearchGenerator,
    model: ProposedImageCaptioningModel,
) -> None:
    """
    The number of generated candidates must match the batch.
    """
    generation_model = model.vit_decoder.model

    assert isinstance(
        generation_model,
        DummyGenerationModel,
    )

    generation_model.candidate_count_delta = -1

    visual_features = torch.randn(
        BATCH_SIZE,
        VIT_TOKENS,
        HIDDEN_SIZE,
    )

    with pytest.raises(RuntimeError):
        generator._generate_branch(
            visual_features=visual_features,
            visual_attention_mask=None,
            decoder=model.vit_decoder,
            branch="vit",
        )


def test_generate_returns_dual_branch_output(
    generator: DualBranchBeamSearchGenerator,
    images: Tensor,
) -> None:
    """
    Full generation must return ViT and Swin candidate groups.
    """
    output = generator.generate(images)

    assert isinstance(
        output,
        DualBeamSearchOutput,
    )

    assert output.vit.branch == "vit"
    assert output.swin.branch == "swin"

    assert len(output.vit.candidates) == BATCH_SIZE
    assert len(output.swin.candidates) == BATCH_SIZE

    for image_candidates in output.vit.candidates:
        assert (
            len(image_candidates)
            == NUM_RETURN_SEQUENCES
        )

    for image_candidates in output.swin.candidates:
        assert (
            len(image_candidates)
            == NUM_RETURN_SEQUENCES
        )


def test_generate_encodes_images_only_once(
    generator: DualBranchBeamSearchGenerator,
    model: ProposedImageCaptioningModel,
    images: Tensor,
) -> None:
    """
    Both branches must reuse one image-encoding operation.
    """
    encoder = model.encoder

    assert isinstance(
        encoder,
        DummyDualVisionEncoder,
    )

    generator.generate(images)

    assert encoder.call_count == 1


def test_generate_creates_different_branch_candidates(
    generator: DualBranchBeamSearchGenerator,
    images: Tensor,
) -> None:
    """
    ViT and Swin outputs must retain their branch identities.
    """
    output = generator.generate(images)

    vit_candidate = output.vit.candidates[0][0]
    swin_candidate = output.swin.candidates[0][0]

    assert vit_candidate.branch == "vit"
    assert swin_candidate.branch == "swin"

    assert (
        vit_candidate.token_ids[1]
        != swin_candidate.token_ids[1]
    )

    assert vit_candidate.text != swin_candidate.text


def test_generate_uses_automatic_masks(
    generator: DualBranchBeamSearchGenerator,
    model: ProposedImageCaptioningModel,
    images: Tensor,
) -> None:
    """
    Full visual masks must reach both generation branches.
    """
    generator.generate(images)

    vit_generation_model = model.vit_decoder.model
    swin_generation_model = model.swin_decoder.model

    assert isinstance(
        vit_generation_model,
        DummyGenerationModel,
    )

    assert isinstance(
        swin_generation_model,
        DummyGenerationModel,
    )

    assert vit_generation_model.last_kwargs is not None
    assert swin_generation_model.last_kwargs is not None

    vit_mask = (
        vit_generation_model
        .last_kwargs["attention_mask"]
    )

    swin_mask = (
        swin_generation_model
        .last_kwargs["attention_mask"]
    )

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


def test_generate_passes_custom_masks(
    generator: DualBranchBeamSearchGenerator,
    model: ProposedImageCaptioningModel,
    images: Tensor,
) -> None:
    """
    Valid custom masks must be forwarded unchanged.
    """
    vit_mask = torch.ones(
        BATCH_SIZE,
        VIT_TOKENS,
        dtype=torch.long,
    )

    vit_mask[:, -1] = 0

    swin_mask = torch.ones(
        BATCH_SIZE,
        SWIN_TOKENS,
        dtype=torch.long,
    )

    swin_mask[:, -1] = 0

    generator.generate(
        images=images,
        vit_visual_attention_mask=vit_mask,
        swin_visual_attention_mask=swin_mask,
    )

    vit_generation_model = model.vit_decoder.model
    swin_generation_model = model.swin_decoder.model

    assert isinstance(
        vit_generation_model,
        DummyGenerationModel,
    )

    assert isinstance(
        swin_generation_model,
        DummyGenerationModel,
    )

    assert vit_generation_model.last_kwargs is not None
    assert swin_generation_model.last_kwargs is not None

    assert (
        vit_generation_model
        .last_kwargs["attention_mask"]
        is vit_mask
    )

    assert (
        swin_generation_model
        .last_kwargs["attention_mask"]
        is swin_mask
    )


def test_generate_can_disable_automatic_masks(
    images: Tensor,
) -> None:
    """
    None must reach generate when automatic masks are disabled.
    """
    model = build_model(
        create_visual_masks=False,
    )

    generator = build_generator(
        model=model,
    )

    generator.generate(images)

    vit_generation_model = model.vit_decoder.model
    swin_generation_model = model.swin_decoder.model

    assert isinstance(
        vit_generation_model,
        DummyGenerationModel,
    )

    assert isinstance(
        swin_generation_model,
        DummyGenerationModel,
    )

    assert vit_generation_model.last_kwargs is not None
    assert swin_generation_model.last_kwargs is not None

    assert (
        vit_generation_model
        .last_kwargs["attention_mask"]
        is None
    )

    assert (
        swin_generation_model
        .last_kwargs["attention_mask"]
        is None
    )


def test_generate_temporarily_uses_eval_and_inference_mode(
    generator: DualBranchBeamSearchGenerator,
    model: ProposedImageCaptioningModel,
    images: Tensor,
) -> None:
    """
    Generation must disable training behavior and gradients.
    """
    model.train()

    generator.generate(images)

    vit_generation_model = model.vit_decoder.model
    swin_generation_model = model.swin_decoder.model

    assert isinstance(
        vit_generation_model,
        DummyGenerationModel,
    )

    assert isinstance(
        swin_generation_model,
        DummyGenerationModel,
    )

    assert (
        vit_generation_model.training_during_generate
        is False
    )

    assert (
        swin_generation_model.training_during_generate
        is False
    )

    assert (
        vit_generation_model.grad_enabled_during_generate
        is False
    )

    assert (
        swin_generation_model.grad_enabled_during_generate
        is False
    )


def test_generate_restores_training_mode(
    generator: DualBranchBeamSearchGenerator,
    model: ProposedImageCaptioningModel,
    images: Tensor,
) -> None:
    """
    A training model must return to training mode afterward.
    """
    model.train()

    generator.generate(images)

    assert model.training is True


def test_generate_preserves_evaluation_mode(
    generator: DualBranchBeamSearchGenerator,
    model: ProposedImageCaptioningModel,
    images: Tensor,
) -> None:
    """
    An evaluation model must remain in evaluation mode.
    """
    model.eval()

    generator.generate(images)

    assert model.training is False


def test_generate_restores_state_after_exception(
    generator: DualBranchBeamSearchGenerator,
    model: ProposedImageCaptioningModel,
    images: Tensor,
) -> None:
    """
    The original state must be restored after generation errors.
    """
    generation_model = model.vit_decoder.model

    assert isinstance(
        generation_model,
        DummyGenerationModel,
    )

    generation_model.raise_generation_error = True

    model.train()

    with pytest.raises(
        RuntimeError,
        match="Intentional dummy generation error",
    ):
        generator.generate(images)

    assert model.training is True