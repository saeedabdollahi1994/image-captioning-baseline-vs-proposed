"""
Projection layers for the proposed image-captioning model.

ViT and Swin may produce feature embeddings with different
hidden dimensions. These projection layers convert both outputs
to the hidden dimension required by the T5 decoder.
"""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class BaseVisionProjection(nn.Module, ABC):
    """
    Abstract base class for visual feature projection layers.

    Every projection layer must receive visual features and
    return features with the decoder hidden dimension.
    """

    @abstractmethod
    def forward(self, features: Tensor) -> Tensor:
        """
        Project visual features to the decoder hidden dimension.

        Args:
            features:
                Visual features with shape:

                    (batch_size, number_of_tokens, input_dim)

        Returns:
            Projected features with shape:

                (batch_size, number_of_tokens, output_dim)
        """


class VisionProjection(BaseVisionProjection):
    """
    Project visual feature embeddings to the T5 hidden dimension.

    This class can be used independently for either ViT or Swin.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize the projection layer.

        Args:
            input_dim:
                Hidden dimension produced by the vision encoder.

            output_dim:
                Hidden dimension expected by the T5 decoder.

            dropout:
                Dropout probability applied after projection.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout

        self.linear = nn.Linear(
            in_features=input_dim,
            out_features= output_dim
        )

        self.normalization = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        """
        Project visual features.

        Args:
            features:
                Tensor with shape:

                    (batch_size, number_of_tokens, input_dim)

        Returns:
            Projected tensor with shape:

                (batch_size, number_of_tokens, output_dim)
        """

        if features.ndim != 3:
            raise ValueError(
                f"Expected a 3D tensor, but received "
                f"{features.ndim} dimensions."
            )

        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension {self.input_dim}, "
                f"but received {features.shape[-1]}."
            )
        
        features = self.linear(features)
        feature_normalized = self.normalization(features)
        projected_feature = self.dropout(feature_normalized)
        return projected_feature

class DualVisionProjection(nn.Module):
    """
    Project ViT and Swin features separately.

    ViT features
        |
        └── ViT projection
                |
                └── T5 hidden dimension

    Swin features
        |
        └── Swin projection
                |
                └── T5 hidden dimension
    """

    def __init__(
        self,
        vit_input_dim: int,
        swin_input_dim: int,
        decoder_hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize both projection branches.

        Args:
            vit_input_dim:
                Hidden dimension produced by ViT.

            swin_input_dim:
                Hidden dimension produced by Swin.

            decoder_hidden_dim:
                Hidden dimension expected by the T5 decoder.

            dropout:
                Dropout probability for both projection branches.
        """
        super().__init__()

        self.vit_input_dim = vit_input_dim
        self.swin_input_dim = swin_input_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.dropout_rate = dropout


        self.vit_projection = VisionProjection(
            input_dim = vit_input_dim,
            output_dim = decoder_hidden_dim,
            dropout=self.dropout_rate
        )

        self.swin_projection = VisionProjection(
            input_dim = swin_input_dim,
            output_dim = decoder_hidden_dim,
            dropout=self.dropout_rate
        )


    def forward(
        self,
        vit_features: Tensor,
        swin_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Project ViT and Swin features.

        Args:
            vit_features:
                ViT tensor with shape:

                    (batch_size, vit_tokens, vit_input_dim)

            swin_features:
                Swin tensor with shape:

                    (batch_size, swin_tokens, swin_input_dim)

        Returns:
            A tuple containing:

                projected_vit_features,
                projected_swin_features

            Both outputs have decoder_hidden_dim as their
            final dimension.
        """

        projected_vit = self.vit_projection(vit_features)
        projected_swin = self.swin_projection(swin_features)

        return projected_vit, projected_swin
        