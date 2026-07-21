"""
T5 decoder for the proposed image-captioning model.

The decoder receives projected visual features instead of
textual encoder features and predicts caption tokens.

The visual features must already have the same hidden dimension
as the selected T5 model.
"""

from abc import ABC, abstractmethod

from torch import Tensor, nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)


class BaseCaptionDecoder(nn.Module, ABC):
    """
    Abstract base class for caption decoders.

    Every caption decoder must receive visual features and
    return token prediction outputs.
    """

    @abstractmethod
    def forward(
        self,
        visual_features: Tensor,
        visual_attention_mask: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> Seq2SeqLMOutput:
        """
        Decode visual features into caption token predictions.

        Args:
            visual_features:
                Projected visual features with shape:

                    (
                        batch_size,
                        number_of_visual_tokens,
                        decoder_hidden_dim,
                    )

            visual_attention_mask:
                Mask for visual tokens with shape:

                    (batch_size, number_of_visual_tokens)

                Values:
                    1 = valid visual token
                    0 = ignored visual token

            decoder_input_ids:
                Caption tokens already given to the decoder.

                Shape:

                    (batch_size, caption_length)

            labels:
                Target caption token IDs used during training.

                Shape:

                    (batch_size, caption_length)

        Returns:
            T5 output containing prediction logits and
            optionally the training loss.
        """
        raise NotImplementedError


class T5CaptionDecoder(BaseCaptionDecoder):
    """
    Generate caption token predictions using a pretrained T5 model.

    Suggested pretrained model:

        t5-small

    The textual T5 encoder is bypassed. Projected visual features
    are passed directly to the T5 decoder as encoder outputs.
    """

    def __init__(
        self,
        model_name: str = "t5-small",
        freeze_decoder: bool = False,
    ) -> None:
        """
        Initialize the T5 caption decoder.

        Args:
            model_name:
                Name of the pretrained T5 model.

            freeze_decoder:
                If True, all T5 parameters will be frozen.
        """
        super().__init__()

        self.model_name = model_name
        self.freeze_decoder = freeze_decoder

        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name
        )

        self.hidden_size = self.model.config.d_model
        self.vocab_size = self.model.config.vocab_size

        if self.freeze_decoder:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        visual_features: Tensor,
        visual_attention_mask: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> Seq2SeqLMOutput:
        """
        Predict caption tokens from projected visual features.

        Args:
            visual_features:
                Projected ViT or Swin features with shape:

                    (
                        batch_size,
                        number_of_visual_tokens,
                        hidden_size,
                    )

            visual_attention_mask:
                Mask indicating valid visual tokens.

                Shape:

                    (batch_size, number_of_visual_tokens)

            decoder_input_ids:
                Input caption token IDs used by the decoder.

                Shape:

                    (batch_size, caption_length)

            labels:
                Target caption token IDs used to calculate loss.

                Shape:

                    (batch_size, caption_length)

        Returns:
            Seq2SeqLMOutput containing prediction logits
            and optionally the training loss.
        """
        if visual_features.ndim != 3:
            raise ValueError(
                f"visual_features must be a 3D tensor, "
                f"but received {visual_features.ndim} dimensions."
            )

        if visual_features.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected visual feature dimension "
                f"{self.hidden_size}, but received "
                f"{visual_features.shape[-1]}."
            )

        if visual_attention_mask is not None:
            if visual_attention_mask.ndim != 2:
                raise ValueError(
                    "visual_attention_mask must be a 2D tensor."
                )

            if visual_attention_mask.shape != visual_features.shape[:2]:
                raise ValueError(
                    "visual_attention_mask shape must match "
                    "the batch size and number of visual tokens. "
                    f"Expected {visual_features.shape[:2]}, "
                    f"but received {visual_attention_mask.shape}."
                )

        if decoder_input_ids is not None:
            if decoder_input_ids.ndim != 2:
                raise ValueError(
                    "decoder_input_ids must be a 2D tensor."
                )

            if decoder_input_ids.shape[0] != visual_features.shape[0]:
                raise ValueError(
                    "decoder_input_ids batch size must match "
                    "visual_features batch size."
                )

        if labels is not None:
            if labels.ndim != 2:
                raise ValueError(
                    "labels must be a 2D tensor."
                )

            if labels.shape[0] != visual_features.shape[0]:
                raise ValueError(
                    "labels batch size must match "
                    "visual_features batch size."
                )

        if labels is None and decoder_input_ids is None:
            raise ValueError(
                "Either labels or decoder_input_ids must be provided."
            )

        encoder_outputs = BaseModelOutput(
            last_hidden_state=visual_features,
        )

        outputs = self.model(
            encoder_outputs=encoder_outputs,
            attention_mask=visual_attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True,
        )

        return outputs