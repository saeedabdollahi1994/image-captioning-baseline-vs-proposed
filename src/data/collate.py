"""Batch collation utilities for training and evaluation datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, TypeAlias

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from .dataset import EvaluationSample, TrainingSample


PaddingStrategy: TypeAlias = Literal["longest", "max_length"]


@dataclass
class TrainingBatch:
    """A tokenized image-caption batch used during training."""

    images: Tensor
    labels: Tensor
    caption_attention_mask: Tensor
    captions: list[str]
    image_names: list[str]

    def to(
        self,
        device: torch.device | str,
        non_blocking: bool = False,
    ) -> "TrainingBatch":
        """Return a new batch whose Tensor fields are on device."""
        if not isinstance(non_blocking, bool):
            raise TypeError(
                "non_blocking must be a boolean, "
                f"not {type(non_blocking).__name__}."
            )

        resolved_device = torch.device(device)

        return TrainingBatch(
            images=self.images.to(
                device=resolved_device,
                non_blocking=non_blocking,
            ),
            labels=self.labels.to(
                device=resolved_device,
                non_blocking=non_blocking,
            ),
            caption_attention_mask=(
                self.caption_attention_mask.to(
                    device=resolved_device,
                    non_blocking=non_blocking,
                )
            ),
            captions=self.captions,
            image_names=self.image_names,
        )


@dataclass
class EvaluationBatch:
    """An image batch together with untokenized references."""

    images: Tensor
    reference_captions: list[list[str]]
    image_names: list[str]

    def to(
        self,
        device: torch.device | str,
        non_blocking: bool = False,
    ) -> "EvaluationBatch":
        """Return a new batch whose image Tensor is on device."""
        if not isinstance(non_blocking, bool):
            raise TypeError(
                "non_blocking must be a boolean, "
                f"not {type(non_blocking).__name__}."
            )

        resolved_device = torch.device(device)

        return EvaluationBatch(
            images=self.images.to(
                device=resolved_device,
                non_blocking=non_blocking,
            ),
            reference_captions=self.reference_captions,
            image_names=self.image_names,
        )


def _validate_image_tensor(
    image: Tensor,
    sample_index: int,
) -> None:
    """Validate one transformed RGB image shaped (3, H, W)."""
    if not torch.is_tensor(image):
        raise TypeError(
            f"Image at sample index {sample_index} must be a "
            f"torch.Tensor, not {type(image).__name__}."
        )

    if image.ndim != 3:
        raise ValueError(
            f"Image at sample index {sample_index} must be a 3D "
            "Tensor shaped (channels, height, width), "
            f"not {tuple(image.shape)}."
        )

    if image.shape[0] != 3:
        raise ValueError(
            f"Image at sample index {sample_index} must have three "
            f"RGB channels, not shape {tuple(image.shape)}."
        )

    if image.shape[1] <= 0 or image.shape[2] <= 0:
        raise ValueError(
            f"Image at sample index {sample_index} must have positive "
            f"height and width, not shape {tuple(image.shape)}."
        )


def _stack_images(
    images: Sequence[Tensor],
) -> Tensor:
    """Validate and stack images into a (B, 3, H, W) Tensor."""
    if len(images) == 0:
        raise ValueError(
            "Cannot stack an empty image collection."
        )

    first_image = images[0]

    _validate_image_tensor(
        image=first_image,
        sample_index=0,
    )

    expected_shape = first_image.shape
    expected_dtype = first_image.dtype
    expected_device = first_image.device

    for sample_index, image in enumerate(images):
        _validate_image_tensor(
            image=image,
            sample_index=sample_index,
        )

        if image.shape != expected_shape:
            raise ValueError(
                f"Image at sample index {sample_index} has shape "
                f"{tuple(image.shape)}; expected "
                f"{tuple(expected_shape)}."
            )

        if image.dtype != expected_dtype:
            raise TypeError(
                f"Image at sample index {sample_index} has dtype "
                f"{image.dtype}; expected {expected_dtype}."
            )

        if image.device != expected_device:
            raise ValueError(
                f"Image at sample index {sample_index} is on "
                f"{image.device}; expected {expected_device}."
            )

    return torch.stack(
        list(images),
        dim=0,
    )


class TrainingCollator:
    """Stack training images and tokenize captions."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 40,
        padding: PaddingStrategy = "longest",
        truncation: bool = True,
        label_pad_token_id: int = -100,
    ) -> None:
        if not isinstance(
            tokenizer,
            PreTrainedTokenizerBase,
        ):
            raise TypeError(
                "tokenizer must inherit from "
                "PreTrainedTokenizerBase, "
                f"not {type(tokenizer).__name__}."
            )

        if tokenizer.pad_token_id is None:
            raise ValueError(
                "tokenizer.pad_token_id must be defined "
                "for batching."
            )

        if isinstance(max_length, bool) or not isinstance(
            max_length,
            int,
        ):
            raise TypeError(
                "max_length must be an integer, "
                f"not {type(max_length).__name__}."
            )

        if max_length <= 0:
            raise ValueError(
                "max_length must be greater than zero, "
                f"not {max_length}."
            )

        if not isinstance(padding, str):
            raise TypeError(
                "padding must be a string, "
                f"not {type(padding).__name__}."
            )

        if padding not in {
            "longest",
            "max_length",
        }:
            raise ValueError(
                "padding must be either 'longest' or "
                f"'max_length', not {padding!r}."
            )

        if not isinstance(truncation, bool):
            raise TypeError(
                "truncation must be a boolean, "
                f"not {type(truncation).__name__}."
            )

        if isinstance(
            label_pad_token_id,
            bool,
        ) or not isinstance(
            label_pad_token_id,
            int,
        ):
            raise TypeError(
                "label_pad_token_id must be an integer, "
                f"not {type(label_pad_token_id).__name__}."
            )

        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_length: int = max_length
        self.padding: PaddingStrategy = padding
        self.truncation: bool = truncation
        self.label_pad_token_id: int = (
            label_pad_token_id
        )

    @staticmethod
    def _validate_training_sample(
        sample: TrainingSample,
        sample_index: int,
    ) -> None:
        """Validate one image-caption training sample."""
        if not isinstance(sample, tuple):
            raise TypeError(
                f"Training sample at index {sample_index} "
                "must be a tuple, "
                f"not {type(sample).__name__}."
            )

        if len(sample) != 3:
            raise ValueError(
                f"Training sample at index {sample_index} "
                "must contain exactly three items, "
                f"not {len(sample)}."
            )

        image, caption, image_name = sample

        _validate_image_tensor(
            image=image,
            sample_index=sample_index,
        )

        if not isinstance(caption, str):
            raise TypeError(
                f"Caption at sample index {sample_index} "
                "must be a string, "
                f"not {type(caption).__name__}."
            )

        if not caption.strip():
            raise ValueError(
                f"Caption at sample index {sample_index} "
                "cannot be empty."
            )

        if not isinstance(image_name, str):
            raise TypeError(
                f"Image name at sample index {sample_index} "
                "must be a string, "
                f"not {type(image_name).__name__}."
            )

        if not image_name.strip():
            raise ValueError(
                f"Image name at sample index {sample_index} "
                "cannot be empty."
            )

    def __call__(
        self,
        batch: Sequence[TrainingSample],
    ) -> TrainingBatch:
        """Convert image-caption samples into a training batch."""
        if len(batch) == 0:
            raise ValueError(
                "Training batch cannot be empty."
            )

        images: list[Tensor] = []
        captions: list[str] = []
        image_names: list[str] = []

        for sample_index, sample in enumerate(batch):
            self._validate_training_sample(
                sample=sample,
                sample_index=sample_index,
            )

            image, caption, image_name = sample

            images.append(image)
            captions.append(caption)
            image_names.append(image_name)

        image_batch = _stack_images(images)

        tokenized = self.tokenizer(
            captions,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        if "input_ids" not in tokenized:
            raise KeyError(
                "Tokenizer output does not contain "
                "'input_ids'."
            )

        input_ids = tokenized["input_ids"]

        if not torch.is_tensor(input_ids):
            raise TypeError(
                "tokenizer['input_ids'] must be a "
                "torch.Tensor, "
                f"not {type(input_ids).__name__}."
            )

        if input_ids.ndim != 2:
            raise ValueError(
                "tokenizer['input_ids'] must be a "
                "2D Tensor, "
                f"not shape {tuple(input_ids.shape)}."
            )

        if "attention_mask" not in tokenized:
            raise KeyError(
                "Tokenizer output does not contain "
                "'attention_mask'."
            )

        caption_attention_mask = tokenized[
            "attention_mask"
        ]

        if not torch.is_tensor(
            caption_attention_mask
        ):
            raise TypeError(
                "tokenizer['attention_mask'] must be "
                "a torch.Tensor, "
                f"not "
                f"{type(caption_attention_mask).__name__}."
            )

        if caption_attention_mask.ndim != 2:
            raise ValueError(
                "tokenizer['attention_mask'] must be "
                "a 2D Tensor, "
                f"not shape "
                f"{tuple(caption_attention_mask.shape)}."
            )

        if (
            caption_attention_mask.shape
            != input_ids.shape
        ):
            raise ValueError(
                "Tokenizer input_ids and attention_mask "
                "must have the same shape. "
                f"Received {tuple(input_ids.shape)} and "
                f"{tuple(caption_attention_mask.shape)}."
            )

        if input_ids.shape[0] != len(batch):
            raise RuntimeError(
                "Tokenizer returned an unexpected "
                "batch size. "
                f"Expected {len(batch)}, "
                f"received {input_ids.shape[0]}."
            )

        labels = input_ids.clone()

        labels[
            caption_attention_mask == 0
        ] = self.label_pad_token_id

        return TrainingBatch(
            images=image_batch,
            labels=labels,
            caption_attention_mask=(
                caption_attention_mask
            ),
            captions=captions,
            image_names=image_names,
        )


class EvaluationCollator:
    """Stack evaluation images and preserve references."""

    @staticmethod
    def _validate_evaluation_sample(
        sample: EvaluationSample,
        sample_index: int,
    ) -> None:
        """Validate one image-reference evaluation sample."""
        if not isinstance(sample, tuple):
            raise TypeError(
                f"Evaluation sample at index "
                f"{sample_index} must be a tuple, "
                f"not {type(sample).__name__}."
            )

        if len(sample) != 3:
            raise ValueError(
                f"Evaluation sample at index "
                f"{sample_index} must contain exactly "
                f"three items, not {len(sample)}."
            )

        image, reference_captions, image_name = sample

        _validate_image_tensor(
            image=image,
            sample_index=sample_index,
        )

        if not isinstance(
            reference_captions,
            list,
        ):
            raise TypeError(
                "Reference captions at sample index "
                f"{sample_index} must be a list, "
                f"not "
                f"{type(reference_captions).__name__}."
            )

        if not reference_captions:
            raise ValueError(
                "Reference captions at sample index "
                f"{sample_index} cannot be empty."
            )

        for caption_index, caption in enumerate(
            reference_captions
        ):
            if not isinstance(caption, str):
                raise TypeError(
                    "Reference caption at sample index "
                    f"{sample_index}, caption index "
                    f"{caption_index} must be a string, "
                    f"not {type(caption).__name__}."
                )

            if not caption.strip():
                raise ValueError(
                    "Reference caption at sample index "
                    f"{sample_index}, caption index "
                    f"{caption_index} cannot be empty."
                )

        if not isinstance(image_name, str):
            raise TypeError(
                f"Image name at sample index "
                f"{sample_index} must be a string, "
                f"not {type(image_name).__name__}."
            )

        if not image_name.strip():
            raise ValueError(
                f"Image name at sample index "
                f"{sample_index} cannot be empty."
            )

    def __call__(
        self,
        batch: Sequence[EvaluationSample],
    ) -> EvaluationBatch:
        """Convert image-reference samples into an evaluation batch."""
        if len(batch) == 0:
            raise ValueError(
                "Evaluation batch cannot be empty."
            )

        images: list[Tensor] = []
        reference_captions: list[list[str]] = []
        image_names: list[str] = []

        for sample_index, sample in enumerate(batch):
            self._validate_evaluation_sample(
                sample=sample,
                sample_index=sample_index,
            )

            image, captions, image_name = sample

            images.append(image)
            reference_captions.append(
                list(captions)
            )
            image_names.append(image_name)

        return EvaluationBatch(
            images=_stack_images(images),
            reference_captions=reference_captions,
            image_names=image_names,
        )