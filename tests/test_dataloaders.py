from __future__ import annotations

import json
import random
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import PreTrainedTokenizerBase

from src.data.collate import (
    EvaluationBatch,
    EvaluationCollator,
    TrainingBatch,
    TrainingCollator,
)
from src.data.dataloaders import (
    DataLoaderSettings,
    ImageCaptionDataLoaders,
    ImageCaptionDatasets,
    _build_loader_kwargs,
    _make_generator,
    _resolve_pin_memory,
    _seed_worker,
    create_dataloaders,
    create_datasets,
)
from src.data.dataset import (
    ImageCaptionEvaluationDataset,
    ImageCaptionTrainingDataset,
)


def train_transform(image: Image.Image) -> Tensor:
    assert image.mode == "RGB"
    return torch.ones(3, 8, 8, dtype=torch.float32)


def evaluation_transform(image: Image.Image) -> Tensor:
    assert image.mode == "RGB"
    return torch.full(
        (3, 8, 8),
        fill_value=2.0,
        dtype=torch.float32,
    )


def make_tokenizer(
    *,
    pad_token_id: int | None = 0,
) -> MagicMock:
    tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    tokenizer.pad_token_id = pad_token_id

    def tokenize(
        captions: list[str],
        *,
        padding: str,
        truncation: bool,
        max_length: int,
        return_attention_mask: bool,
        return_tensors: str,
    ) -> dict[str, Tensor]:
        assert return_attention_mask is True
        assert return_tensors == "pt"

        sequences: list[list[int]] = []

        for caption in captions:
            token_ids = list(
                range(
                    2,
                    2 + len(caption.split()),
                )
            )
            token_ids.append(1)

            if truncation:
                token_ids = token_ids[:max_length]

            sequences.append(token_ids)

        if padding == "longest":
            padded_length = max(
                len(sequence)
                for sequence in sequences
            )
        else:
            padded_length = max_length

        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []

        for sequence in sequences:
            sequence = sequence[:padded_length]
            padding_length = padded_length - len(sequence)

            input_ids.append(
                sequence
                + [pad_token_id or 0] * padding_length
            )
            attention_masks.append(
                [1] * len(sequence)
                + [0] * padding_length
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
        }

    tokenizer.side_effect = tokenize
    return tokenizer


@pytest.fixture
def data_paths(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    image_names = [
        "train_1.jpg",
        "train_2.jpg",
        "train_3.jpg",
        "val_1.jpg",
        "test_1.jpg",
    ]

    for index, image_name in enumerate(image_names):
        image = Image.new(
            "RGB",
            size=(12, 10),
            color=(index * 20, 50, 100),
        )
        image.save(images_dir / image_name)

    captions_path = tmp_path / "captions.txt"
    captions_path.write_text(
        "\n".join(
            [
                "image,caption",
                "train_1.jpg,a dog runs",
                "train_1.jpg,a brown dog runs outside",
                "train_2.jpg,two children play",
                "train_2.jpg,children play in the park",
                "train_3.jpg,a cyclist rides",
                "train_3.jpg,a person rides a bicycle",
                "val_1.jpg,a cat sleeps",
                "val_1.jpg,a small cat is sleeping",
                "test_1.jpg,a bird flies",
                "test_1.jpg,a bird is flying in the sky",
            ]
        ),
        encoding="utf-8",
    )

    split_path = tmp_path / "shared_split.json"
    split_path.write_text(
        json.dumps(
            {
                "train_images": [
                    "train_1.jpg",
                    "train_2.jpg",
                    "train_3.jpg",
                ],
                "val_images": ["val_1.jpg"],
                "test_images": ["test_1.jpg"],
            }
        ),
        encoding="utf-8",
    )

    return images_dir, captions_path, split_path


def make_settings(**changes: object) -> DataLoaderSettings:
    return replace(
        DataLoaderSettings(),
        **changes,
    )


def test_settings_defaults_are_valid() -> None:
    settings = DataLoaderSettings()

    settings.validate()

    assert settings.train_batch_size == 16
    assert settings.evaluation_batch_size == 16
    assert settings.num_workers == 0
    assert settings.pin_memory is None
    assert settings.persistent_workers is False
    assert settings.prefetch_factor == 2
    assert settings.shuffle_train is True
    assert settings.drop_last_train is False
    assert settings.seed == 42
    assert settings.max_caption_length == 40
    assert settings.padding == "longest"
    assert settings.truncation is True
    assert settings.label_pad_token_id == -100


def test_settings_accept_valid_custom_values() -> None:
    settings = DataLoaderSettings(
        train_batch_size=8,
        evaluation_batch_size=4,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,
        shuffle_train=False,
        drop_last_train=True,
        seed=7,
        max_caption_length=25,
        padding="max_length",
        truncation=False,
        label_pad_token_id=-5,
    )

    settings.validate()


@pytest.mark.parametrize(
    "field_name",
    [
        "train_batch_size",
        "evaluation_batch_size",
        "prefetch_factor",
        "max_caption_length",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [0, -1],
)
def test_settings_reject_non_positive_integer_values(
    field_name: str,
    invalid_value: int,
) -> None:
    settings = make_settings(
        **{field_name: invalid_value}
    )

    with pytest.raises(ValueError, match=field_name):
        settings.validate()


@pytest.mark.parametrize(
    "field_name",
    [
        "train_batch_size",
        "evaluation_batch_size",
        "prefetch_factor",
        "max_caption_length",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [True, 1.5, "2", None],
)
def test_settings_reject_invalid_positive_integer_types(
    field_name: str,
    invalid_value: object,
) -> None:
    settings = make_settings(
        **{field_name: invalid_value}
    )

    with pytest.raises(TypeError, match=field_name):
        settings.validate()


@pytest.mark.parametrize(
    "field_name",
    ["num_workers", "seed"],
)
def test_settings_reject_negative_non_negative_values(
    field_name: str,
) -> None:
    settings = make_settings(
        **{field_name: -1}
    )

    with pytest.raises(ValueError, match=field_name):
        settings.validate()


@pytest.mark.parametrize(
    "field_name",
    ["num_workers", "seed"],
)
@pytest.mark.parametrize(
    "invalid_value",
    [True, 1.5, "2", None],
)
def test_settings_reject_invalid_non_negative_integer_types(
    field_name: str,
    invalid_value: object,
) -> None:
    settings = make_settings(
        **{field_name: invalid_value}
    )

    with pytest.raises(TypeError, match=field_name):
        settings.validate()


@pytest.mark.parametrize(
    "invalid_value",
    [0, 1, "true", object()],
)
def test_settings_reject_invalid_pin_memory(
    invalid_value: object,
) -> None:
    settings = make_settings(
        pin_memory=invalid_value,
    )

    with pytest.raises(TypeError, match="pin_memory"):
        settings.validate()


@pytest.mark.parametrize(
    "field_name",
    [
        "persistent_workers",
        "shuffle_train",
        "drop_last_train",
        "truncation",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [0, 1, "true", None],
)
def test_settings_reject_invalid_boolean_fields(
    field_name: str,
    invalid_value: object,
) -> None:
    settings = make_settings(
        **{field_name: invalid_value}
    )

    with pytest.raises(TypeError, match=field_name):
        settings.validate()


def test_settings_reject_persistent_workers_without_workers() -> None:
    settings = make_settings(
        num_workers=0,
        persistent_workers=True,
    )

    with pytest.raises(
        ValueError,
        match="persistent_workers",
    ):
        settings.validate()


@pytest.mark.parametrize(
    "invalid_padding",
    ["shortest", "", None, 1],
)
def test_settings_reject_invalid_padding(
    invalid_padding: object,
) -> None:
    settings = make_settings(
        padding=invalid_padding,
    )

    with pytest.raises(ValueError, match="padding"):
        settings.validate()


@pytest.mark.parametrize(
    "invalid_value",
    [True, 1.5, "-100", None],
)
def test_settings_reject_invalid_label_pad_token_id(
    invalid_value: object,
) -> None:
    settings = make_settings(
        label_pad_token_id=invalid_value,
    )

    with pytest.raises(
        TypeError,
        match="label_pad_token_id",
    ):
        settings.validate()


def test_settings_are_frozen() -> None:
    settings = DataLoaderSettings()

    with pytest.raises(FrozenInstanceError):
        settings.train_batch_size = 32  # type: ignore[misc]


def test_make_generator_is_deterministic() -> None:
    first = torch.rand(
        5,
        generator=_make_generator(123),
    )
    second = torch.rand(
        5,
        generator=_make_generator(123),
    )

    assert torch.equal(first, second)


def test_make_generator_changes_with_seed() -> None:
    first = torch.rand(
        5,
        generator=_make_generator(123),
    )
    second = torch.rand(
        5,
        generator=_make_generator(456),
    )

    assert not torch.equal(first, second)


def test_seed_worker_seeds_python_random(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_mock = MagicMock()
    monkeypatch.setattr(
        random,
        "seed",
        seed_mock,
    )
    monkeypatch.setattr(
        torch,
        "initial_seed",
        lambda: 2**32 + 123,
    )

    _seed_worker(4)

    seed_mock.assert_called_once_with(123)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False)],
)
def test_resolve_pin_memory_preserves_explicit_value(
    value: bool,
    expected: bool,
) -> None:
    assert _resolve_pin_memory(value) is expected


def test_resolve_pin_memory_uses_cuda_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: True,
    )

    assert _resolve_pin_memory(None) is True

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    assert _resolve_pin_memory(None) is False


def test_build_loader_kwargs_without_workers() -> None:
    collator = object()

    kwargs = _build_loader_kwargs(
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        shuffle=True,
        drop_last=False,
        seed=11,
        collate_fn=collator,
    )

    assert kwargs["batch_size"] == 4
    assert kwargs["shuffle"] is True
    assert kwargs["num_workers"] == 0
    assert kwargs["pin_memory"] is False
    assert kwargs["drop_last"] is False
    assert kwargs["collate_fn"] is collator
    assert isinstance(kwargs["generator"], torch.Generator)
    assert "persistent_workers" not in kwargs
    assert "prefetch_factor" not in kwargs
    assert "worker_init_fn" not in kwargs


def test_build_loader_kwargs_with_workers() -> None:
    collator = object()

    kwargs = _build_loader_kwargs(
        batch_size=4,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,
        shuffle=False,
        drop_last=True,
        seed=11,
        collate_fn=collator,
    )

    assert kwargs["num_workers"] == 2
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 3
    assert kwargs["worker_init_fn"] is _seed_worker


def test_create_datasets_builds_expected_dataset_types(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    datasets = create_datasets(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
    )

    assert isinstance(datasets, ImageCaptionDatasets)
    assert isinstance(
        datasets.train,
        ImageCaptionTrainingDataset,
    )
    assert isinstance(
        datasets.validation_loss,
        ImageCaptionTrainingDataset,
    )
    assert isinstance(
        datasets.validation_metrics,
        ImageCaptionEvaluationDataset,
    )
    assert isinstance(
        datasets.test,
        ImageCaptionEvaluationDataset,
    )


def test_create_datasets_uses_expected_splits_and_lengths(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    datasets = create_datasets(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
    )

    assert datasets.train.split_name == "train"
    assert datasets.validation_loss.split_name == "val"
    assert datasets.validation_metrics.split_name == "val"
    assert datasets.test.split_name == "test"

    assert len(datasets.train) == 6
    assert len(datasets.validation_loss) == 2
    assert len(datasets.validation_metrics) == 1
    assert len(datasets.test) == 1


def test_create_datasets_assigns_expected_transforms(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    datasets = create_datasets(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
    )

    assert datasets.train.transform is train_transform
    assert (
        datasets.validation_loss.transform
        is evaluation_transform
    )
    assert (
        datasets.validation_metrics.transform
        is evaluation_transform
    )
    assert datasets.test.transform is evaluation_transform


def test_create_dataloaders_builds_four_loaders(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        settings=DataLoaderSettings(
            train_batch_size=2,
            evaluation_batch_size=1,
        ),
    )

    assert isinstance(loaders, ImageCaptionDataLoaders)
    assert isinstance(loaders.train, DataLoader)
    assert isinstance(loaders.validation_loss, DataLoader)
    assert isinstance(loaders.validation_metrics, DataLoader)
    assert isinstance(loaders.test, DataLoader)


def test_create_dataloaders_uses_expected_datasets(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
    )

    assert isinstance(
        loaders.train.dataset,
        ImageCaptionTrainingDataset,
    )
    assert isinstance(
        loaders.validation_loss.dataset,
        ImageCaptionTrainingDataset,
    )
    assert isinstance(
        loaders.validation_metrics.dataset,
        ImageCaptionEvaluationDataset,
    )
    assert isinstance(
        loaders.test.dataset,
        ImageCaptionEvaluationDataset,
    )

    assert loaders.train.dataset.split_name == "train"
    assert loaders.validation_loss.dataset.split_name == "val"
    assert loaders.validation_metrics.dataset.split_name == "val"
    assert loaders.test.dataset.split_name == "test"


def test_create_dataloaders_applies_batch_sizes(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        settings=DataLoaderSettings(
            train_batch_size=3,
            evaluation_batch_size=2,
        ),
    )

    assert loaders.train.batch_size == 3
    assert loaders.validation_loss.batch_size == 2
    assert loaders.validation_metrics.batch_size == 2
    assert loaders.test.batch_size == 2


def test_create_dataloaders_assigns_expected_collators(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
    )

    assert isinstance(
        loaders.train.collate_fn,
        TrainingCollator,
    )
    assert (
        loaders.validation_loss.collate_fn
        is loaders.train.collate_fn
    )
    assert isinstance(
        loaders.validation_metrics.collate_fn,
        EvaluationCollator,
    )
    assert isinstance(
        loaders.test.collate_fn,
        EvaluationCollator,
    )
    assert (
        loaders.validation_metrics.collate_fn
        is loaders.test.collate_fn
    )


def test_create_dataloaders_passes_tokenization_settings(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths
    tokenizer = make_tokenizer()

    loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=tokenizer,
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        settings=DataLoaderSettings(
            max_caption_length=17,
            padding="max_length",
            truncation=False,
            label_pad_token_id=-9,
        ),
    )

    collator = loaders.train.collate_fn

    assert isinstance(collator, TrainingCollator)
    assert collator.tokenizer is tokenizer
    assert collator.max_length == 17
    assert collator.padding == "max_length"
    assert collator.truncation is False
    assert collator.label_pad_token_id == -9


def test_train_loader_uses_random_sampler_when_shuffling(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        settings=DataLoaderSettings(
            shuffle_train=True,
        ),
    )

    assert isinstance(loaders.train.sampler, RandomSampler)


def test_train_loader_uses_sequential_sampler_when_not_shuffling(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        settings=DataLoaderSettings(
            shuffle_train=False,
        ),
    )

    assert isinstance(
        loaders.train.sampler,
        SequentialSampler,
    )


def test_evaluation_loaders_always_use_sequential_sampler(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
    )

    assert isinstance(
        loaders.validation_loss.sampler,
        SequentialSampler,
    )
    assert isinstance(
        loaders.validation_metrics.sampler,
        SequentialSampler,
    )
    assert isinstance(loaders.test.sampler, SequentialSampler)


def test_drop_last_is_applied_only_to_train_loader(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        settings=DataLoaderSettings(
            drop_last_train=True,
        ),
    )

    assert loaders.train.drop_last is True
    assert loaders.validation_loss.drop_last is False
    assert loaders.validation_metrics.drop_last is False
    assert loaders.test.drop_last is False


def test_num_workers_and_pin_memory_are_applied(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        settings=DataLoaderSettings(
            num_workers=0,
            pin_memory=True,
        ),
    )

    for loader in (
        loaders.train,
        loaders.validation_loss,
        loaders.validation_metrics,
        loaders.test,
    ):
        assert loader.num_workers == 0
        assert loader.pin_memory is True


def test_worker_specific_options_are_applied(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        settings=DataLoaderSettings(
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=3,
        ),
    )

    for loader in (
        loaders.train,
        loaders.validation_loss,
        loaders.validation_metrics,
        loaders.test,
    ):
        assert loader.num_workers == 2
        assert loader.persistent_workers is True
        assert loader.prefetch_factor == 3
        assert loader.worker_init_fn is _seed_worker


def test_default_pin_memory_follows_cuda_availability(
    data_paths: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images_dir, captions_path, split_path = data_paths
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: True,
    )

    loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
    )

    assert loaders.train.pin_memory is True
    assert loaders.validation_loss.pin_memory is True
    assert loaders.validation_metrics.pin_memory is True
    assert loaders.test.pin_memory is True


def test_train_loader_returns_training_batch(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        settings=DataLoaderSettings(
            train_batch_size=2,
            shuffle_train=False,
        ),
    )

    batch = next(iter(loaders.train))

    assert isinstance(batch, TrainingBatch)
    assert batch.images.shape == (2, 3, 8, 8)
    assert torch.all(batch.images == 1.0)
    assert batch.labels.ndim == 2
    assert batch.caption_attention_mask.shape == batch.labels.shape
    assert batch.captions == [
        "a dog runs",
        "a brown dog runs outside",
    ]
    assert batch.image_names == [
        "train_1.jpg",
        "train_1.jpg",
    ]
    assert torch.all(
        batch.labels[
            batch.caption_attention_mask == 0
        ]
        == -100
    )


def test_validation_loss_loader_returns_training_batch(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        settings=DataLoaderSettings(
            evaluation_batch_size=2,
        ),
    )

    batch = next(iter(loaders.validation_loss))

    assert isinstance(batch, TrainingBatch)
    assert batch.images.shape == (2, 3, 8, 8)
    assert torch.all(batch.images == 2.0)
    assert batch.image_names == [
        "val_1.jpg",
        "val_1.jpg",
    ]


def test_validation_metrics_loader_returns_evaluation_batch(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        settings=DataLoaderSettings(
            evaluation_batch_size=1,
        ),
    )

    batch = next(iter(loaders.validation_metrics))

    assert isinstance(batch, EvaluationBatch)
    assert batch.images.shape == (1, 3, 8, 8)
    assert torch.all(batch.images == 2.0)
    assert batch.reference_captions == [
        [
            "a cat sleeps",
            "a small cat is sleeping",
        ]
    ]
    assert batch.image_names == ["val_1.jpg"]


def test_test_loader_returns_evaluation_batch(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        settings=DataLoaderSettings(
            evaluation_batch_size=1,
        ),
    )

    batch = next(iter(loaders.test))

    assert isinstance(batch, EvaluationBatch)
    assert batch.images.shape == (1, 3, 8, 8)
    assert torch.all(batch.images == 2.0)
    assert batch.reference_captions == [
        [
            "a bird flies",
            "a bird is flying in the sky",
        ]
    ]
    assert batch.image_names == ["test_1.jpg"]


def test_drop_last_changes_train_loader_length(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    keep_last = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        settings=DataLoaderSettings(
            train_batch_size=4,
            drop_last_train=False,
        ),
    )
    drop_last = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        settings=DataLoaderSettings(
            train_batch_size=4,
            drop_last_train=True,
        ),
    )

    assert len(keep_last.train) == 2
    assert len(drop_last.train) == 1


def test_train_shuffle_is_reproducible_with_same_seed(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths
    settings = DataLoaderSettings(
        train_batch_size=2,
        shuffle_train=True,
        seed=91,
    )

    first_loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        settings=settings,
    )
    second_loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        settings=settings,
    )

    first_order = [
        image_name
        for batch in first_loaders.train
        for image_name in batch.image_names
    ]
    second_order = [
        image_name
        for batch in second_loaders.train
        for image_name in batch.image_names
    ]

    assert first_order == second_order


def test_create_dataloaders_uses_default_settings(
    data_paths: tuple[Path, Path, Path],
) -> None:
    images_dir, captions_path, split_path = data_paths

    loaders = create_dataloaders(
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
        tokenizer=make_tokenizer(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        settings=None,
    )

    assert loaders.train.batch_size == 16
    assert loaders.validation_loss.batch_size == 16
    assert loaders.validation_metrics.batch_size == 16
    assert loaders.test.batch_size == 16


def test_create_dataloaders_validates_settings_before_data_access(
    tmp_path: Path,
) -> None:
    settings = DataLoaderSettings(
        train_batch_size=0,
    )

    with pytest.raises(
        ValueError,
        match="train_batch_size",
    ):
        create_dataloaders(
            images_dir=tmp_path / "missing-images",
            captions_path=tmp_path / "missing-captions.txt",
            split_path=tmp_path / "missing-split.json",
            tokenizer=make_tokenizer(),
            train_transform=train_transform,
            evaluation_transform=evaluation_transform,
            settings=settings,
        )