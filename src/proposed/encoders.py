"""
Vision encoders for the proposed image-captioning model.

This file contains two encoders:

1. ViTEncoder
2. SwinEncoder

Each encoder receives a batch of images and returns
visual feature embeddings.
"""
from torch import Tensor
from torch import nn
from abc import ABC, abstractmethod
from transformers import  ViTModel, SwinModel

class BaseVisionEncoder(nn.Module, ABC):

    @abstractmethod
    def forward(self, images: Tensor) -> Tensor:
        raise NotImplementedError
    

class ViTEncoder(BaseVisionEncoder):
    """
    Extract visual features using a Vision Transformer.

    Suggested pretrained model:

        google/vit-base-patch16-224
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        freeze_encoder: bool = True,
    ) -> None:
        """
        Initialize the ViT encoder.

        Args:
            model_name:
                Name of the pretrained ViT model.

            freeze_encoder:
                If True, ViT weights will not be updated
                during training.
        """
        super().__init__()

        self.model_name = model_name
        self.freeze_encoder = freeze_encoder

        self.model = ViTModel.from_pretrained(self.model_name)
        self.hidden_size = self.model.config.hidden_size

        if self.freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        Extract ViT feature embeddings.

        Args:
            images:
                Image batch with shape:

                (batch_size, channels, height, width)

        Returns:
            ViT features with shape:

                (batch_size, number_of_tokens, hidden_size)

        Example output shape:

            (8, 197, 768)
        """
        outputs = self.model(pixel_values=images)
        features = outputs.last_hidden_state
        return features

        


class SwinEncoder(BaseVisionEncoder):
    """
    Extract visual features using a Swin Transformer.

    Suggested pretrained model:

        microsoft/swin-base-patch4-window7-224
    """

    def __init__(
        self,
        model_name: str = "microsoft/swin-base-patch4-window7-224",
        freeze_encoder: bool = True,
    ) -> None:
        """
        Initialize the Swin encoder.

        Args:
            model_name:
                Name of the pretrained Swin model.

            freeze_encoder:
                If True, Swin weights will not be updated
                during training.
        """
        super().__init__()

        self.model_name = model_name
        self.freeze_encoder = freeze_encoder

        self.model = SwinModel.from_pretrained(self.model_name)
        self.hidden_size = self.model.config.hidden_size

        if self.freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False


    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        Extract Swin feature embeddings.

        Args:
            images:
                Image batch with shape:

                (batch_size, channels, height, width)

        Returns:
            Swin features with shape:

                (batch_size, number_of_tokens, hidden_size)
        """

        outputs = self.model(pixel_values=images)
        features = outputs.last_hidden_state
        return features


class DualVisionEncoder(nn.Module):
    """
    Run ViT and Swin encoders in parallel.

    Image batch
        |
        |---- ViT encoder
        |
        |---- Swin encoder

    The class returns the features of both branches.
    """

    def __init__(
        self,
        vit_model_name: str = "google/vit-base-patch16-224",
        swin_model_name: str = "microsoft/swin-base-patch4-window7-224",
        freeze_encoders: bool = True,
    ) -> None:
        """
        Initialize both visual encoders.

        Args:
            vit_model_name:
                Name of the pretrained ViT model.

            swin_model_name:
                Name of the pretrained Swin model.

            freeze_encoders:
                If True, both encoders will be frozen.
        """
        super().__init__()

        self.vit_encoder = ViTEncoder(
            model_name=vit_model_name,
            freeze_encoder=freeze_encoders,
        )

        self.swin_encoder = SwinEncoder(
            model_name=swin_model_name,
            freeze_encoder=freeze_encoders,
        )

    def forward(
        self,
        images: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Extract visual features from both encoders.

        Args:
            images:
                Image batch with shape:

                (batch_size, channels, height, width)

        Returns:
            A tuple containing:

                vit_features,
                swin_features
        """

        vit_features = self.vit_encoder(images)
        swin_features = self.swin_encoder(images)

        return vit_features,swin_features