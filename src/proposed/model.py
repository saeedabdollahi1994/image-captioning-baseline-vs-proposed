"""
This module connects:

    image
        -> ViT and Swin encoders
        -> separate projection layers
        -> separate T5 caption decoders
        -> caption logits and optional training losses

Beam search and ensemble reranking are intentionally not implemented
in this file. They belong to beam_search.py and ensemble.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import Seq2SeqLMOutput

from .encoders import DualVisionEncoder
from .projection import DualVisionProjection

from .t5_decoder import T5CaptionDecoder


@dataclass
class DualCaptionModelOutput:
    """
    Store the outputs of the two caption-generation branches.

    Attributes:
        vit_outputs:
            Complete T5 output generated from the ViT branch.

            Important fields:
                vit_outputs.loss
                vit_outputs.logits

        swin_outputs:
            Complete T5 output generated from the Swin branch.

            Important fields:
                swin_outputs.loss
                swin_outputs.logits
    """

    vit_outputs: Seq2SeqLMOutput
    swin_outputs: Seq2SeqLMOutput


class BaseImageCaptioningModel(nn.Module, ABC):
    """
    Abstract contract for image-captioning models.

    Every image-captioning model must be able to:

        1. Convert images into decoder-compatible visual features.
        2. Use those visual features to predict caption tokens.
    """

    @abstractmethod
    def encode_images(
        self,
        images: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Encode images and project their visual features.

        Args:
            images:
                Image batch with shape:

                    (
                        batch_size,
                        channels,
                        height,
                        width,
                    )

                Expected for the selected ViT and Swin models:

                    (batch_size, 3, 224, 224)

        Returns:
            A tuple containing:

                projected_vit_features
                projected_swin_features

            Expected shapes:

                projected_vit_features:
                    (
                        batch_size,
                        vit_visual_tokens,
                        t5_hidden_size,
                    )

                projected_swin_features:
                    (
                        batch_size,
                        swin_visual_tokens,
                        t5_hidden_size,
                    )
        """
        

    @abstractmethod
    def forward(
        self,
        images: Tensor,
        labels: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        vit_visual_attention_mask: Tensor | None = None,
        swin_visual_attention_mask: Tensor | None = None,
    ) -> DualCaptionModelOutput:
        """
        Generate caption predictions from images.

        Args:
            images:
                Image batch with shape:

                    (batch_size, 3, height, width)

            labels:
                Target caption token IDs used during training.

                Shape:

                    (batch_size, caption_length)

                Padding positions should contain -100.

            decoder_input_ids:
                Caption tokens already given to the decoder.

                Used mainly during manual autoregressive decoding
                and beam search.

                Shape:

                    (batch_size, current_caption_length)

            vit_visual_attention_mask:
                Mask for projected ViT tokens.

                Shape:

                    (
                        batch_size,
                        vit_visual_tokens,
                    )

            swin_visual_attention_mask:
                Mask for projected Swin tokens.

                Shape:

                    (
                        batch_size,
                        swin_visual_tokens,
                    )

        Returns:
            Outputs from both ViT-T5 and Swin-T5 branches.
        """
        


class ProposedImageCaptioningModel(BaseImageCaptioningModel):
    """
    Proposed dual-encoder image-captioning model.

    Architecture:

        Images
          |
          +-- ViT Encoder
          |      |
          |      +-- ViT Projection
          |              |
          |              +-- ViT T5 Decoder
          |
          +-- Swin Encoder
                 |
                 +-- Swin Projection
                         |
                         +-- Swin T5 Decoder

    The two branches produce independent caption predictions.
    Their candidates will later be combined and reranked in
    ensemble.py.
    """

    def __init__(
        self,
        encoder: DualVisionEncoder,
        projection: DualVisionProjection,
        vit_decoder: T5CaptionDecoder,
        swin_decoder: T5CaptionDecoder,
        create_visual_masks: bool = True,
    ) -> None:
        """
        Initialize the complete proposed model.

        Args:
            encoder:
                Dual encoder containing ViT and Swin models.

            projection:
                Dual projection module that converts both encoder
                feature dimensions to the corresponding T5 hidden
                dimensions.

            vit_decoder:
                T5 decoder used by the ViT branch.

            swin_decoder:
                T5 decoder used by the Swin branch.

            create_visual_masks:
                If True and no mask is passed to forward, create
                an all-ones attention mask automatically.

                This is suitable for the current fixed-resolution
                ViT and Swin outputs because all visual tokens are
                valid.
        """
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.vit_decoder = vit_decoder
        self.swin_decoder = swin_decoder
        self.create_visual_masks = create_visual_masks
        modules = {
            "encoder": encoder,
            "projection": projection,
            "vit_decoder": vit_decoder,
            "swin_decoder": swin_decoder,
            }

        for module_name, module in modules.items():
            if not isinstance(module, nn.Module):
                raise TypeError(
                    f"{module_name} must be an instance of nn.Module, "
                    f"but received {type(module).__name__}."
                )

        if (
            self.projection.vit_projection.output_dim
            != self.vit_decoder.hidden_size
            ):
            raise ValueError(
                "ViT projection output dimension must match "
                "the ViT decoder hidden size. "
                f"Projection output dimension: "
                f"{self.projection.vit_projection.output_dim}. "
                f"Decoder hidden size: "
                f"{self.vit_decoder.hidden_size}."
            )
        
        if (
            self.projection.swin_projection.output_dim
            != self.swin_decoder.hidden_size
            ):
            raise ValueError(
                "Swin projection output dimension must match "
                "the Swin decoder hidden size. "
                f"Projection output dimension: "
                f"{self.projection.swin_projection.output_dim}. "
                f"Decoder hidden size: "
                f"{self.swin_decoder.hidden_size}."
            )



    @staticmethod
    def _validate_images(
        images: Tensor,
    ) -> None:
        """
        Validate the input image batch.

        Args:
            images:
                Expected shape:

                    (
                        batch_size,
                        channels,
                        height,
                        width,
                    )

        Raises:
            TypeError:
                If images is not a Tensor.

            ValueError:
                If images does not have four dimensions.

            ValueError:
                If images does not contain three RGB channels.
        """
        if not isinstance (images,Tensor):
            raise TypeError("Image type should be a tensor.")
        if images.ndim != 4:
            raise ValueError("images must be a 4D tensor with shape ,(batch_size, channels, height, width)."
)
        if images.shape[1] != 3:
            raise ValueError("Image should be in RGB format.")

    @staticmethod
    def _create_full_visual_attention_mask(
        visual_features: Tensor,
    ) -> Tensor:
        """
        Create a mask in which every visual token is valid.

        Args:
            visual_features:
                Projected features with shape:

                    (
                        batch_size,
                        visual_tokens,
                        hidden_size,
                    )

        Returns:
            A mask with shape:

                (
                    batch_size,
                    visual_tokens,
                )

            Every value must be 1.
        """
        visual_mask = torch.ones(
                            visual_features.shape[:2],
                            dtype=torch.long,
                            device=visual_features.device,
                        )
        
        return visual_mask

    @staticmethod
    def _validate_visual_attention_mask(
        mask: Tensor,
        visual_features: Tensor,
        mask_name: str,
    ) -> None:
        """
        Validate one visual attention mask.

        Args:
            mask:
                Visual token mask.

            visual_features:
                Corresponding projected visual features.

            mask_name:
                Human-readable mask name used in error messages.

                Examples:
                    "vit_visual_attention_mask"
                    "swin_visual_attention_mask"
        """
        if mask.ndim != 2:
            raise ValueError(
                f"{mask_name} must be a 2D tensor."
            )

        if mask.shape != visual_features.shape[:2]:
            raise ValueError(
                f"{mask_name} shape must match visual feature shape. "
                f"Expected {tuple(visual_features.shape[:2])}, "
                f"but received {tuple(mask.shape)}."
            )

        if mask.device != visual_features.device:
            raise ValueError(
                f"{mask_name} and visual_features must be "
                "on the same device."
            )


    def encode_images(
        self,
        images: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Encode and project a batch of images.

        Processing order:

            images
                -> DualVisionEncoder
                -> raw ViT and Swin features
                -> DualVisionProjection
                -> T5-compatible visual features

        Args:
            images:
                Image Tensor with shape:

                    (batch_size, 3, height, width)

        Returns:
            A tuple:

                (
                    projected_vit_features,
                    projected_swin_features,
                )
        """
        self._validate_images(images)
        
        vit_features, swin_features  = self.encoder(images)
        projected_vit_features, projected_swin_features= self.projection(vit_features, swin_features)

        if (
            projected_vit_features.shape[-1]
            != self.vit_decoder.hidden_size
        ):
            raise ValueError(
                "The projected ViT feature dimension must match "
                "the ViT decoder hidden size. "
                f"Projected dimension: "
                f"{projected_vit_features.shape[-1]}; "
                f"decoder hidden size: "
                f"{self.vit_decoder.hidden_size}."
            )

        if (
            projected_swin_features.shape[-1]
            != self.swin_decoder.hidden_size
        ):
            raise ValueError(
                "The projected Swin feature dimension must match "
                "the Swin decoder hidden size. "
                f"Projected dimension: "
                f"{projected_swin_features.shape[-1]}; "
                f"decoder hidden size: "
                f"{self.swin_decoder.hidden_size}."
            )

        return projected_vit_features,projected_swin_features

    def forward(
        self,
        images: Tensor,
        labels: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        vit_visual_attention_mask: Tensor | None = None,
        swin_visual_attention_mask: Tensor | None = None,
    ) -> DualCaptionModelOutput:
        """
        Run both image-captioning branches.

        Training usage:

            outputs = model(
                images=images,
                labels=labels,
            )

            vit_loss = outputs.vit_outputs.loss
            swin_loss = outputs.swin_outputs.loss

        Manual decoding usage:

            outputs = model(
                images=images,
                decoder_input_ids=current_token_ids,
            )

            vit_logits = outputs.vit_outputs.logits
            swin_logits = outputs.swin_outputs.logits

        Args:
            images:
                Input image batch.

            labels:
                Correct caption token IDs during training.

            decoder_input_ids:
                Previously generated caption token IDs during
                autoregressive decoding.

            vit_visual_attention_mask:
                Optional mask for ViT visual tokens.

            swin_visual_attention_mask:
                Optional mask for Swin visual tokens.

        Returns:
            DualCaptionModelOutput containing both T5 outputs.
        """
        projected_vit_features, projected_swin_features = self.encode_images(images)
        
        if vit_visual_attention_mask is None:
            if self.create_visual_masks:
                vit_visual_attention_mask = (
                    self._create_full_visual_attention_mask(
                        projected_vit_features
                    )
                )
        else:
            self._validate_visual_attention_mask(
                mask=vit_visual_attention_mask,
                visual_features=projected_vit_features,
                mask_name="vit_visual_attention_mask",
            )

        if swin_visual_attention_mask is None:
            if self.create_visual_masks:
                swin_visual_attention_mask = (
                    self._create_full_visual_attention_mask(
                        projected_swin_features
                    )
                )
        else:
            self._validate_visual_attention_mask(
                mask=swin_visual_attention_mask,
                visual_features=projected_swin_features,
                mask_name="swin_visual_attention_mask",
            )

        vit_outputs = self.vit_decoder(
            visual_features=projected_vit_features,
            visual_attention_mask=vit_visual_attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        swin_outputs = self.swin_decoder(
            visual_features=projected_swin_features,
            visual_attention_mask=swin_visual_attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        return DualCaptionModelOutput(
            vit_outputs=vit_outputs,
            swin_outputs=swin_outputs,
        )