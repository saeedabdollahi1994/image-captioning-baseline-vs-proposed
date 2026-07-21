"""
Beam-search candidate generation for the proposed model.

This module receives projected ViT and Swin visual features and
uses the corresponding T5 decoders to generate multiple caption
candidates for every image.

Responsibilities:

    1. Encode and project images once.
    2. Run beam search independently for ViT and Swin.
    3. Preserve token IDs, decoded captions, and beam scores.
    4. Group candidates by image and model branch.

CLIP scoring, consensus scoring, reranking, and final caption
selection belong to ensemble.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Literal, TypeAlias, cast

import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutput

from .model import ProposedImageCaptioningModel
from .t5_decoder import T5CaptionDecoder


BranchName: TypeAlias = Literal["vit", "swin"]


@dataclass(frozen=True)
class BeamSearchConfig:
    """
    Configuration for deterministic beam-search generation.

    Attributes:
        num_beams:
            Number of active beams maintained during generation.

        num_return_sequences:
            Number of final candidates returned for each image
            and each model branch.

        max_new_tokens:
            Maximum number of newly generated caption tokens.

        min_new_tokens:
            Minimum number of newly generated tokens before EOS
            is allowed to finish the sequence.

        length_penalty:
            Length normalization value used by beam search.

        early_stopping:
            Whether beam search can stop once enough completed
            candidates have been found.

        no_repeat_ngram_size:
            Size of n-grams that may not be repeated.

            Use 0 to disable this restriction.

        use_cache:
            Whether T5 should cache decoder attention states
            during autoregressive generation.
    """

    num_beams: int = 5
    num_return_sequences: int = 5
    max_new_tokens: int = 30
    min_new_tokens: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = True
    no_repeat_ngram_size: int = 2
    use_cache: bool = True

    def validate(self) -> None:
        """
        Validate all beam-search configuration values.

        Raises:
            TypeError:
                If a configuration value has the wrong type.

            ValueError:
                If a numeric value is outside its valid range.
        """
        if isinstance(self.num_beams, bool) or not isinstance(
            self.num_beams,
            int,
        ):
            raise TypeError(
                "num_beams must be an integer, "
                f"but received {type(self.num_beams).__name__}."
            )

        if self.num_beams <= 1:
            raise ValueError(
                "num_beams must be greater than 1 for beam search, "
                f"but received {self.num_beams}."
            )

        if isinstance(
            self.num_return_sequences,
            bool,
        ) or not isinstance(
            self.num_return_sequences,
            int,
        ):
            raise TypeError(
                "num_return_sequences must be an integer, "
                f"but received "
                f"{type(self.num_return_sequences).__name__}."
            )

        if self.num_return_sequences <= 0:
            raise ValueError(
                "num_return_sequences must be greater than 0, "
                f"but received {self.num_return_sequences}."
            )

        if self.num_return_sequences > self.num_beams:
            raise ValueError(
                "num_return_sequences must be less than or equal "
                "to num_beams. "
                f"Received num_return_sequences="
                f"{self.num_return_sequences} and "
                f"num_beams={self.num_beams}."
            )

        if isinstance(self.max_new_tokens, bool) or not isinstance(
            self.max_new_tokens,
            int,
        ):
            raise TypeError(
                "max_new_tokens must be an integer, "
                f"but received "
                f"{type(self.max_new_tokens).__name__}."
            )

        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than 0, "
                f"but received {self.max_new_tokens}."
            )

        if isinstance(self.min_new_tokens, bool) or not isinstance(
            self.min_new_tokens,
            int,
        ):
            raise TypeError(
                "min_new_tokens must be an integer, "
                f"but received "
                f"{type(self.min_new_tokens).__name__}."
            )

        if self.min_new_tokens < 0:
            raise ValueError(
                "min_new_tokens must be greater than or equal "
                f"to 0, but received {self.min_new_tokens}."
            )

        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError(
                "min_new_tokens must be less than or equal to "
                "max_new_tokens. "
                f"Received min_new_tokens={self.min_new_tokens} "
                f"and max_new_tokens={self.max_new_tokens}."
            )

        if isinstance(self.length_penalty, bool) or not isinstance(
            self.length_penalty,
            (int, float),
        ):
            raise TypeError(
                "length_penalty must be an int or float, "
                f"but received "
                f"{type(self.length_penalty).__name__}."
            )

        numeric_length_penalty = float(self.length_penalty)

        if not math.isfinite(numeric_length_penalty):
            raise ValueError(
                "length_penalty must be a finite number, "
                f"but received {self.length_penalty}."
            )

        if numeric_length_penalty <= 0.0:
            raise ValueError(
                "length_penalty must be greater than 0, "
                f"but received {self.length_penalty}."
            )

        if not isinstance(self.early_stopping, bool):
            raise TypeError(
                "early_stopping must be a boolean, "
                f"but received "
                f"{type(self.early_stopping).__name__}."
            )

        if isinstance(
            self.no_repeat_ngram_size,
            bool,
        ) or not isinstance(
            self.no_repeat_ngram_size,
            int,
        ):
            raise TypeError(
                "no_repeat_ngram_size must be an integer, "
                f"but received "
                f"{type(self.no_repeat_ngram_size).__name__}."
            )

        if self.no_repeat_ngram_size < 0:
            raise ValueError(
                "no_repeat_ngram_size must be greater than or "
                f"equal to 0, but received "
                f"{self.no_repeat_ngram_size}."
            )

        if not isinstance(self.use_cache, bool):
            raise TypeError(
                "use_cache must be a boolean, "
                f"but received {type(self.use_cache).__name__}."
            )


@dataclass(frozen=True)
class BeamCandidate:
    """
    Store one generated caption candidate.

    Attributes:
        branch:
            Branch that generated the candidate.

        rank:
            Candidate rank within one image and branch.

        text:
            Decoded caption string.

        token_ids:
            Generated T5 token IDs.

        beam_score:
            Final score returned by beam search.
    """

    branch: BranchName
    rank: int
    text: str
    token_ids: tuple[int, ...]
    beam_score: float


@dataclass(frozen=True)
class BranchBeamSearchOutput:
    """
    Store candidates generated by one model branch.

    candidates is grouped as:

        candidates[image_index][candidate_index]
    """

    branch: BranchName
    candidates: list[list[BeamCandidate]]


@dataclass(frozen=True)
class DualBeamSearchOutput:
    """
    Store beam-search results from both model branches.
    """

    vit: BranchBeamSearchOutput
    swin: BranchBeamSearchOutput


class BaseBeamSearchGenerator(nn.Module, ABC):
    """
    Abstract contract for caption candidate generators.
    """

    @abstractmethod
    def generate(
        self,
        images: Tensor,
        vit_visual_attention_mask: Tensor | None = None,
        swin_visual_attention_mask: Tensor | None = None,
    ) -> DualBeamSearchOutput:
        """
        Generate caption candidates for every input image.
        """
        raise NotImplementedError


class DualBranchBeamSearchGenerator(BaseBeamSearchGenerator):
    """
    Generate caption candidates from ViT-T5 and Swin-T5.

    Processing flow:

        images
          |
          +-- encode_images(...)
          |
          +-- projected ViT features
          |       |
          |       +-- ViT T5 generate(...)
          |
          +-- projected Swin features
                  |
                  +-- Swin T5 generate(...)
    """

    def __init__(
        self,
        model: ProposedImageCaptioningModel,
        tokenizer: PreTrainedTokenizerBase,
        config: BeamSearchConfig | None = None,
    ) -> None:
        """
        Initialize dual-branch beam-search generation.

        Args:
            model:
                Complete proposed image-captioning model.

            tokenizer:
                Tokenizer corresponding to the T5 decoders.

            config:
                Optional beam-search configuration.
        """
        super().__init__()

        if not isinstance(
            model,
            ProposedImageCaptioningModel,
        ):
            raise TypeError(
                "model must be an instance of "
                "ProposedImageCaptioningModel, "
                f"but received {type(model).__name__}."
            )

        if not isinstance(
            tokenizer,
            PreTrainedTokenizerBase,
        ):
            raise TypeError(
                "tokenizer must be an instance of "
                "PreTrainedTokenizerBase, "
                f"but received {type(tokenizer).__name__}."
            )

        if config is None:
            resolved_config = BeamSearchConfig()
        else:
            if not isinstance(config, BeamSearchConfig):
                raise TypeError(
                    "config must be a BeamSearchConfig instance "
                    f"or None, but received "
                    f"{type(config).__name__}."
                )

            resolved_config = config

        resolved_config.validate()

        self.model: ProposedImageCaptioningModel = model
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.config: BeamSearchConfig = resolved_config

        vit_vocab_size = self.model.vit_decoder.vocab_size
        swin_vocab_size = self.model.swin_decoder.vocab_size

        if vit_vocab_size != swin_vocab_size:
            raise ValueError(
                "ViT and Swin decoders must use the same "
                "vocabulary size. "
                f"ViT vocabulary size: {vit_vocab_size}; "
                f"Swin vocabulary size: {swin_vocab_size}."
            )

        tokenizer_vocab_size = len(self.tokenizer)

        if tokenizer_vocab_size > vit_vocab_size:
            raise ValueError(
                "The tokenizer vocabulary cannot be larger than "
                "the decoder vocabulary. "
                f"Tokenizer size: {tokenizer_vocab_size}; "
                f"decoder size: {vit_vocab_size}."
            )

    @staticmethod
    def _validate_branch_name(
        branch: str,
    ) -> BranchName:
        """
        Validate and narrow an encoder branch name.

        Args:
            branch:
                Expected values:
                    "vit"
                    "swin"

        Returns:
            Validated branch name.
        """
        if branch not in {"vit", "swin"}:
            raise ValueError(
                "branch must be either 'vit' or 'swin', "
                f"but received {branch!r}."
            )

        return cast(BranchName, branch)

    def _prepare_visual_attention_mask(
        self,
        visual_features: Tensor,
        visual_attention_mask: Tensor | None,
        mask_name: str,
    ) -> Tensor | None:
        """
        Validate or create one branch's visual attention mask.
        """
        if visual_attention_mask is not None:
            if not torch.is_tensor(visual_attention_mask):
                raise TypeError(
                    f"{mask_name} must be a torch.Tensor, "
                    f"but received "
                    f"{type(visual_attention_mask).__name__}."
                )

            self.model._validate_visual_attention_mask(
                mask=visual_attention_mask,
                visual_features=visual_features,
                mask_name=mask_name,
            )

            return visual_attention_mask

        if self.model.create_visual_masks:
            return self.model._create_full_visual_attention_mask(
                visual_features
            )

        return None

    @staticmethod
    def _wrap_visual_features(
        visual_features: Tensor,
    ) -> BaseModelOutput:
        """
        Wrap projected visual features as T5 encoder outputs.
        """
        if not torch.is_tensor(visual_features):
            raise TypeError(
                "visual_features must be a torch.Tensor, "
                f"but received "
                f"{type(visual_features).__name__}."
            )

        if visual_features.ndim != 3:
            raise ValueError(
                "visual_features must be a 3D tensor with shape "
                "(batch_size, visual_tokens, hidden_size), "
                f"but received shape "
                f"{tuple(visual_features.shape)}."
            )

        return BaseModelOutput(
            last_hidden_state=visual_features,
        )

    def _generate_branch(
        self,
        visual_features: Tensor,
        visual_attention_mask: Tensor | None,
        decoder: T5CaptionDecoder,
        branch: BranchName,
    ) -> BranchBeamSearchOutput:
        """
        Run beam search for one visual encoder branch.
        """
        validated_branch = self._validate_branch_name(branch)

        if not isinstance(decoder, T5CaptionDecoder):
            raise TypeError(
                "decoder must be an instance of "
                "T5CaptionDecoder, "
                f"but received {type(decoder).__name__}."
            )

        if not torch.is_tensor(visual_features):
            raise TypeError(
                "visual_features must be a torch.Tensor."
            )

        if visual_features.ndim != 3:
            raise ValueError(
                "visual_features must be a 3D tensor with shape "
                "(batch_size, visual_tokens, hidden_size)."
            )

        if visual_features.shape[-1] != decoder.hidden_size:
            raise ValueError(
                f"{validated_branch} visual feature dimension "
                "must match its decoder hidden size. "
                f"Feature dimension: "
                f"{visual_features.shape[-1]}; "
                f"decoder hidden size: {decoder.hidden_size}."
            )

        if visual_attention_mask is not None:
            self.model._validate_visual_attention_mask(
                mask=visual_attention_mask,
                visual_features=visual_features,
                mask_name=(
                    f"{validated_branch}_visual_attention_mask"
                ),
            )

        encoder_outputs = self._wrap_visual_features(
            visual_features
        )

        generation_output = decoder.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=visual_attention_mask,
            num_beams=self.config.num_beams,
            num_return_sequences=(
                self.config.num_return_sequences
            ),
            max_new_tokens=self.config.max_new_tokens,
            min_new_tokens=self.config.min_new_tokens,
            length_penalty=self.config.length_penalty,
            early_stopping=self.config.early_stopping,
            no_repeat_ngram_size=(
                self.config.no_repeat_ngram_size
            ),
            do_sample=False,
            use_cache=self.config.use_cache,
            return_dict_in_generate=True,
            output_scores=True,
        )

        sequences = generation_output.sequences

        sequence_scores = getattr(
            generation_output,
            "sequences_scores",
            None,
        )

        if sequence_scores is None:
            raise RuntimeError(
                f"Beam-search sequence scores were not returned "
                f"for the {validated_branch} branch. "
                "Ensure num_beams is greater than 1 and "
                "output_scores=True is passed to generate()."
            )

        if sequences.ndim != 2:
            raise RuntimeError(
                "Generated sequences must be a 2D tensor, "
                f"but received shape {tuple(sequences.shape)}."
            )

        if sequence_scores.ndim != 1:
            raise RuntimeError(
                "Sequence scores must be a 1D tensor, "
                f"but received shape "
                f"{tuple(sequence_scores.shape)}."
            )

        batch_size = visual_features.shape[0]

        expected_candidate_count = (
            batch_size
            * self.config.num_return_sequences
        )

        if sequences.shape[0] != expected_candidate_count:
            raise RuntimeError(
                "Unexpected number of generated sequences. "
                f"Expected {expected_candidate_count}, "
                f"but received {sequences.shape[0]}."
            )

        if sequence_scores.shape[0] != expected_candidate_count:
            raise RuntimeError(
                "Unexpected number of beam scores. "
                f"Expected {expected_candidate_count}, "
                f"but received {sequence_scores.shape[0]}."
            )

        decoded_texts = self.tokenizer.batch_decode(
            sequences.detach().cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        if len(decoded_texts) != expected_candidate_count:
            raise RuntimeError(
                "The number of decoded captions does not match "
                "the number of generated sequences. "
                f"Expected {expected_candidate_count}, "
                f"but received {len(decoded_texts)}."
            )

        grouped_candidates: list[list[BeamCandidate]] = []

        for image_index in range(batch_size):
            image_candidates: list[BeamCandidate] = []

            for candidate_index in range(
                self.config.num_return_sequences
            ):
                flat_index = (
                    image_index
                    * self.config.num_return_sequences
                    + candidate_index
                )

                token_ids = tuple(
                    int(token_id)
                    for token_id in (
                        sequences[flat_index]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                )

                beam_score = float(
                    sequence_scores[flat_index]
                    .detach()
                    .cpu()
                    .item()
                )

                candidate = BeamCandidate(
                    branch=validated_branch,
                    rank=candidate_index + 1,
                    text=decoded_texts[flat_index].strip(),
                    token_ids=token_ids,
                    beam_score=beam_score,
                )

                image_candidates.append(candidate)

            sorted_candidates = sorted(
                image_candidates,
                key=lambda candidate: candidate.beam_score,
                reverse=True,
            )

            ranked_candidates = [
                BeamCandidate(
                    branch=candidate.branch,
                    rank=rank,
                    text=candidate.text,
                    token_ids=candidate.token_ids,
                    beam_score=candidate.beam_score,
                )
                for rank, candidate in enumerate(
                    sorted_candidates,
                    start=1,
                )
            ]

            grouped_candidates.append(ranked_candidates)

        return BranchBeamSearchOutput(
            branch=validated_branch,
            candidates=grouped_candidates,
        )

    def generate(
        self,
        images: Tensor,
        vit_visual_attention_mask: Tensor | None = None,
        swin_visual_attention_mask: Tensor | None = None,
    ) -> DualBeamSearchOutput:
        """
        Generate caption candidates from both model branches.

        The images are encoded only once. Beam search is then
        executed independently for the ViT and Swin branches.

        The model's original training/evaluation state is restored
        even when generation raises an exception.
        """
        was_training = self.model.training

        self.model.eval()

        try:
            with torch.inference_mode():
                (
                    projected_vit_features,
                    projected_swin_features,
                ) = self.model.encode_images(images)

                prepared_vit_mask = (
                    self._prepare_visual_attention_mask(
                        visual_features=projected_vit_features,
                        visual_attention_mask=(
                            vit_visual_attention_mask
                        ),
                        mask_name=(
                            "vit_visual_attention_mask"
                        ),
                    )
                )

                prepared_swin_mask = (
                    self._prepare_visual_attention_mask(
                        visual_features=projected_swin_features,
                        visual_attention_mask=(
                            swin_visual_attention_mask
                        ),
                        mask_name=(
                            "swin_visual_attention_mask"
                        ),
                    )
                )

                vit_output = self._generate_branch(
                    visual_features=projected_vit_features,
                    visual_attention_mask=prepared_vit_mask,
                    decoder=self.model.vit_decoder,
                    branch="vit",
                )

                swin_output = self._generate_branch(
                    visual_features=projected_swin_features,
                    visual_attention_mask=prepared_swin_mask,
                    decoder=self.model.swin_decoder,
                    branch="swin",
                )

                dual_output = DualBeamSearchOutput(
                    vit=vit_output,
                    swin=swin_output,
                )

        finally:
            self.model.train(was_training)

        return dual_output