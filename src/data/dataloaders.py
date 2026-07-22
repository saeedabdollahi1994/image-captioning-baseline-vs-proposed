"""Factories for image-captioning datasets and data loaders."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from .collate import (
    EvaluationCollator,
    PaddingStrategy,
    TrainingCollator,
)
from .dataset import (
    ImageCaptionEvaluationDataset,
    ImageCaptionTrainingDataset,
    ImageTransform,
)


@dataclass(frozen=True, slots=True)
class DataLoaderSettings:
    """Configuration shared by all image-captioning data loaders."""

    train_batch_size: int = 16
    evaluation_batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool | None = None
    persistent_workers: bool = False
    prefetch_factor: int = 2
    shuffle_train: bool = True
    drop_last_train: bool = False
    seed: int = 42
    max_caption_length: int = 40
    padding: PaddingStrategy = "longest"
    truncation: bool = True
    label_pad_token_id: int = -100

    def validate(self) -> None:
        """Validate data-loader and caption-tokenization settings."""
        _validate_positive_integer(self.train_batch_size, "train_batch_size")
        _validate_positive_integer(
            self.evaluation_batch_size,
            "evaluation_batch_size",
        )
        _validate_non_negative_integer(self.num_workers, "num_workers")
        _validate_positive_integer(self.prefetch_factor, "prefetch_factor")
        _validate_non_negative_integer(self.seed, "seed")
        _validate_positive_integer(
            self.max_caption_length,
            "max_caption_length",
        )

        if self.pin_memory is not None and not isinstance(
            self.pin_memory,
            bool,
        ):
            raise TypeError(
                "pin_memory must be a boolean or None, "
                f"not {type(self.pin_memory).__name__}."
            )

        for name, value in (
            ("persistent_workers", self.persistent_workers),
            ("shuffle_train", self.shuffle_train),
            ("drop_last_train", self.drop_last_train),
            ("truncation", self.truncation),
        ):
            if not isinstance(value, bool):
                raise TypeError(
                    f"{name} must be a boolean, "
                    f"not {type(value).__name__}."
                )

        if self.persistent_workers and self.num_workers == 0:
            raise ValueError(
                "persistent_workers requires num_workers greater "
                "than zero."
            )

        if self.padding not in {"longest", "max_length"}:
            raise ValueError(
                "padding must be either 'longest' or "
                f"'max_length', not {self.padding!r}."
            )

        if isinstance(
            self.label_pad_token_id,
            bool,
        ) or not isinstance(
            self.label_pad_token_id,
            int,
        ):
            raise TypeError(
                "label_pad_token_id must be an integer, "
                f"not {type(self.label_pad_token_id).__name__}."
            )


@dataclass(frozen=True, slots=True)
class ImageCaptionDatasets:
    """Datasets needed for training, validation, and testing."""

    train: ImageCaptionTrainingDataset
    validation_loss: ImageCaptionTrainingDataset
    validation_metrics: ImageCaptionEvaluationDataset
    test: ImageCaptionEvaluationDataset


@dataclass(frozen=True, slots=True)
class ImageCaptionDataLoaders:
    """Data loaders needed for training, validation, and testing."""

    train: DataLoader
    validation_loss: DataLoader
    validation_metrics: DataLoader
    test: DataLoader


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name} must be an integer, "
            f"not {type(value).__name__}."
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than zero, not {value}."
        )


def _validate_non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name} must be an integer, "
            f"not {type(value).__name__}."
        )

    if value < 0:
        raise ValueError(
            f"{name} must be zero or greater, not {value}."
        )


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def _make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _resolve_pin_memory(pin_memory: bool | None) -> bool:
    if pin_memory is None:
        return torch.cuda.is_available()

    return pin_memory


def _build_loader_kwargs(
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    shuffle: bool,
    drop_last: bool,
    seed: int,
    collate_fn: Any,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
        "generator": _make_generator(seed),
    }

    if num_workers > 0:
        kwargs.update(
            {
                "persistent_workers": persistent_workers,
                "prefetch_factor": prefetch_factor,
                "worker_init_fn": _seed_worker,
            }
        )

    return kwargs


def create_datasets(
    *,
    images_dir: str | Path,
    captions_path: str | Path,
    split_path: str | Path,
    train_transform: ImageTransform,
    evaluation_transform: ImageTransform,
) -> ImageCaptionDatasets:
    """Create all datasets required by the training pipeline."""
    train_dataset = ImageCaptionTrainingDataset(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        split_name="train",
        transform=train_transform,
    )

    validation_loss_dataset = ImageCaptionTrainingDataset(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        split_name="val",
        transform=evaluation_transform,
    )

    validation_metrics_dataset = ImageCaptionEvaluationDataset(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        split_name="val",
        transform=evaluation_transform,
    )

    test_dataset = ImageCaptionEvaluationDataset(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        split_name="test",
        transform=evaluation_transform,
    )

    return ImageCaptionDatasets(
        train=train_dataset,
        validation_loss=validation_loss_dataset,
        validation_metrics=validation_metrics_dataset,
        test=test_dataset,
    )


def create_dataloaders(
    *,
    images_dir: str | Path,
    captions_path: str | Path,
    split_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    train_transform: ImageTransform,
    evaluation_transform: ImageTransform,
    settings: DataLoaderSettings | None = None,
) -> ImageCaptionDataLoaders:
    """Create training, validation, and test data loaders."""
    resolved_settings = settings or DataLoaderSettings()
    resolved_settings.validate()

    datasets = create_datasets(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
    )

    training_collator = TrainingCollator(
        tokenizer=tokenizer,
        max_length=resolved_settings.max_caption_length,
        padding=resolved_settings.padding,
        truncation=resolved_settings.truncation,
        label_pad_token_id=resolved_settings.label_pad_token_id,
    )
    evaluation_collator = EvaluationCollator()

    pin_memory = _resolve_pin_memory(resolved_settings.pin_memory)

    train_loader = DataLoader(
        datasets.train,
        **_build_loader_kwargs(
            batch_size=resolved_settings.train_batch_size,
            num_workers=resolved_settings.num_workers,
            pin_memory=pin_memory,
            persistent_workers=resolved_settings.persistent_workers,
            prefetch_factor=resolved_settings.prefetch_factor,
            shuffle=resolved_settings.shuffle_train,
            drop_last=resolved_settings.drop_last_train,
            seed=resolved_settings.seed,
            collate_fn=training_collator,
        ),
    )

    validation_loss_loader = DataLoader(
        datasets.validation_loss,
        **_build_loader_kwargs(
            batch_size=resolved_settings.evaluation_batch_size,
            num_workers=resolved_settings.num_workers,
            pin_memory=pin_memory,
            persistent_workers=resolved_settings.persistent_workers,
            prefetch_factor=resolved_settings.prefetch_factor,
            shuffle=False,
            drop_last=False,
            seed=resolved_settings.seed + 1,
            collate_fn=training_collator,
        ),
    )

    validation_metrics_loader = DataLoader(
        datasets.validation_metrics,
        **_build_loader_kwargs(
            batch_size=resolved_settings.evaluation_batch_size,
            num_workers=resolved_settings.num_workers,
            pin_memory=pin_memory,
            persistent_workers=resolved_settings.persistent_workers,
            prefetch_factor=resolved_settings.prefetch_factor,
            shuffle=False,
            drop_last=False,
            seed=resolved_settings.seed + 2,
            collate_fn=evaluation_collator,
        ),
    )

    test_loader = DataLoader(
        datasets.test,
        **_build_loader_kwargs(
            batch_size=resolved_settings.evaluation_batch_size,
            num_workers=resolved_settings.num_workers,
            pin_memory=pin_memory,
            persistent_workers=resolved_settings.persistent_workers,
            prefetch_factor=resolved_settings.prefetch_factor,
            shuffle=False,
            drop_last=False,
            seed=resolved_settings.seed + 3,
            collate_fn=evaluation_collator,
        ),
    )

    return ImageCaptionDataLoaders(
        train=train_loader,
        validation_loss=validation_loss_loader,
        validation_metrics=validation_metrics_loader,
        test=test_loader,
    )