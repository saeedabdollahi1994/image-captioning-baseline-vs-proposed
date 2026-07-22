from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from src.data.collate import (
    EvaluationBatch,
    EvaluationCollator,
    TrainingBatch,
    TrainingCollator,
    _stack_images,
    _validate_image_tensor,
)


def make_image(
    value: float = 0.0,
    shape: tuple[int, int, int] = (3, 8, 8),
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> Tensor:
    return torch.full(
        shape,
        fill_value=value,
        dtype=dtype,
        device=device,
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
            padded_length = max(len(sequence) for sequence in sequences)
        else:
            padded_length = max_length

        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []

        for sequence in sequences:
            sequence = sequence[:padded_length]
            padding_length = padded_length - len(sequence)

            input_ids.append(
                sequence + [pad_token_id or 0] * padding_length
            )
            attention_masks.append(
                [1] * len(sequence) + [0] * padding_length
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


def make_training_collator(
    **kwargs: object,
) -> TrainingCollator:
    tokenizer = kwargs.pop("tokenizer", make_tokenizer())

    return TrainingCollator(
        tokenizer=tokenizer,
        **kwargs,
    )


def make_training_batch_input() -> list[tuple[Tensor, str, str]]:
    return [
        (
            make_image(value=1.0),
            "a dog runs",
            "dog.jpg",
        ),
        (
            make_image(value=2.0),
            "two children play outside",
            "children.jpg",
        ),
    ]


def make_evaluation_batch_input() -> list[tuple[Tensor, list[str], str]]:
    return [
        (
            make_image(value=1.0),
            [
                "a dog runs",
                "a dog is running outside",
            ],
            "dog.jpg",
        ),
        (
            make_image(value=2.0),
            [
                "two children play",
                "children are playing outside",
            ],
            "children.jpg",
        ),
    ]


def test_validate_image_tensor_accepts_rgb_tensor() -> None:
    _validate_image_tensor(
        image=make_image(),
        sample_index=0,
    )


def test_validate_image_tensor_rejects_non_tensor() -> None:
    with pytest.raises(TypeError, match="sample index 3"):
        _validate_image_tensor(
            image="not-a-tensor",  # type: ignore[arg-type]
            sample_index=3,
        )


def test_validate_image_tensor_rejects_two_dimensions() -> None:
    with pytest.raises(ValueError, match="3D Tensor"):
        _validate_image_tensor(
            image=torch.zeros(8, 8),
            sample_index=0,
        )


def test_validate_image_tensor_rejects_four_dimensions() -> None:
    with pytest.raises(ValueError, match="3D Tensor"):
        _validate_image_tensor(
            image=torch.zeros(1, 3, 8, 8),
            sample_index=0,
        )


def test_validate_image_tensor_rejects_non_rgb_channels() -> None:
    with pytest.raises(ValueError, match="three RGB channels"):
        _validate_image_tensor(
            image=torch.zeros(1, 8, 8),
            sample_index=0,
        )


def test_validate_image_tensor_rejects_zero_height() -> None:
    with pytest.raises(ValueError, match="positive height and width"):
        _validate_image_tensor(
            image=torch.empty(3, 0, 8),
            sample_index=0,
        )


def test_validate_image_tensor_rejects_zero_width() -> None:
    with pytest.raises(ValueError, match="positive height and width"):
        _validate_image_tensor(
            image=torch.empty(3, 8, 0),
            sample_index=0,
        )


def test_stack_images_returns_batched_tensor() -> None:
    first = make_image(value=1.0)
    second = make_image(value=2.0)

    result = _stack_images([first, second])

    assert result.shape == (2, 3, 8, 8)
    assert result.dtype == torch.float32
    assert torch.equal(result[0], first)
    assert torch.equal(result[1], second)


def test_stack_images_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="empty image collection"):
        _stack_images([])


def test_stack_images_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="expected"):
        _stack_images(
            [
                make_image(shape=(3, 8, 8)),
                make_image(shape=(3, 10, 8)),
            ]
        )


def test_stack_images_rejects_mismatched_dtypes() -> None:
    with pytest.raises(TypeError, match="dtype"):
        _stack_images(
            [
                make_image(dtype=torch.float32),
                make_image(dtype=torch.float64),
            ]
        )


def test_stack_images_rejects_mismatched_devices() -> None:
    with pytest.raises(ValueError, match="expected"):
        _stack_images(
            [
                make_image(device="cpu"),
                make_image(device="meta"),
            ]
        )


def test_training_batch_to_cpu_returns_new_batch() -> None:
    batch = TrainingBatch(
        images=torch.ones(2, 3, 8, 8),
        labels=torch.ones(2, 4, dtype=torch.long),
        caption_attention_mask=torch.ones(
            2,
            4,
            dtype=torch.long,
        ),
        captions=["caption one", "caption two"],
        image_names=["one.jpg", "two.jpg"],
    )

    moved = batch.to("cpu")

    assert moved is not batch
    assert moved.images.device.type == "cpu"
    assert moved.labels.device.type == "cpu"
    assert moved.caption_attention_mask.device.type == "cpu"
    assert moved.captions is batch.captions
    assert moved.image_names is batch.image_names


def test_training_batch_to_rejects_invalid_non_blocking() -> None:
    batch = TrainingBatch(
        images=torch.ones(1, 3, 8, 8),
        labels=torch.ones(1, 2, dtype=torch.long),
        caption_attention_mask=torch.ones(1, 2, dtype=torch.long),
        captions=["caption"],
        image_names=["image.jpg"],
    )

    with pytest.raises(TypeError, match="non_blocking"):
        batch.to(
            "cpu",
            non_blocking=1,  # type: ignore[arg-type]
        )


def test_evaluation_batch_to_cpu_returns_new_batch() -> None:
    references = [["caption one", "caption two"]]
    image_names = ["image.jpg"]

    batch = EvaluationBatch(
        images=torch.ones(1, 3, 8, 8),
        reference_captions=references,
        image_names=image_names,
    )

    moved = batch.to("cpu")

    assert moved is not batch
    assert moved.images.device.type == "cpu"
    assert moved.reference_captions is references
    assert moved.image_names is image_names


def test_evaluation_batch_to_rejects_invalid_non_blocking() -> None:
    batch = EvaluationBatch(
        images=torch.ones(1, 3, 8, 8),
        reference_captions=[["caption"]],
        image_names=["image.jpg"],
    )

    with pytest.raises(TypeError, match="non_blocking"):
        batch.to(
            "cpu",
            non_blocking="yes",  # type: ignore[arg-type]
        )


def test_training_collator_stores_valid_configuration() -> None:
    tokenizer = make_tokenizer()

    collator = TrainingCollator(
        tokenizer=tokenizer,
        max_length=25,
        padding="max_length",
        truncation=False,
        label_pad_token_id=-50,
    )

    assert collator.tokenizer is tokenizer
    assert collator.max_length == 25
    assert collator.padding == "max_length"
    assert collator.truncation is False
    assert collator.label_pad_token_id == -50


def test_training_collator_rejects_invalid_tokenizer() -> None:
    with pytest.raises(TypeError, match="tokenizer"):
        TrainingCollator(
            tokenizer=object(),  # type: ignore[arg-type]
        )


def test_training_collator_rejects_missing_pad_token_id() -> None:
    with pytest.raises(ValueError, match="pad_token_id"):
        TrainingCollator(
            tokenizer=make_tokenizer(
                pad_token_id=None,
            )
        )


@pytest.mark.parametrize(
    "invalid_max_length",
    [True, 4.5, "40", None],
)
def test_training_collator_rejects_non_integer_max_length(
    invalid_max_length: object,
) -> None:
    with pytest.raises(TypeError, match="max_length"):
        make_training_collator(
            max_length=invalid_max_length,
        )


@pytest.mark.parametrize(
    "invalid_max_length",
    [0, -1, -40],
)
def test_training_collator_rejects_non_positive_max_length(
    invalid_max_length: int,
) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        make_training_collator(
            max_length=invalid_max_length,
        )


def test_training_collator_rejects_non_string_padding() -> None:
    with pytest.raises(TypeError, match="padding"):
        make_training_collator(
            padding=1,
        )


def test_training_collator_rejects_unknown_padding() -> None:
    with pytest.raises(ValueError, match="max_length"):
        make_training_collator(
            padding="left",
        )


def test_training_collator_rejects_non_boolean_truncation() -> None:
    with pytest.raises(TypeError, match="truncation"):
        make_training_collator(
            truncation=1,
        )


@pytest.mark.parametrize(
    "invalid_label_pad_token_id",
    [True, 1.5, "-100", None],
)
def test_training_collator_rejects_invalid_label_pad_token_id(
    invalid_label_pad_token_id: object,
) -> None:
    with pytest.raises(TypeError, match="label_pad_token_id"):
        make_training_collator(
            label_pad_token_id=invalid_label_pad_token_id,
        )


def test_training_collator_builds_expected_batch() -> None:
    tokenizer = make_tokenizer()
    collator = make_training_collator(
        tokenizer=tokenizer,
        max_length=10,
        padding="longest",
    )

    result = collator(make_training_batch_input())

    assert isinstance(result, TrainingBatch)
    assert result.images.shape == (2, 3, 8, 8)
    assert result.labels.shape == (2, 5)
    assert result.caption_attention_mask.shape == (2, 5)
    assert result.captions == [
        "a dog runs",
        "two children play outside",
    ]
    assert result.image_names == [
        "dog.jpg",
        "children.jpg",
    ]

    assert tokenizer.call_count == 1

    call_args, call_kwargs = tokenizer.call_args

    assert call_args == (
        [
            "a dog runs",
            "two children play outside",
        ],
    )

    assert call_kwargs == {
        "padding": "longest",
        "truncation": True,
        "max_length": 10,
        "return_attention_mask": True,
        "return_tensors": "pt",
    }


def test_training_collator_replaces_padding_in_labels() -> None:
    collator = make_training_collator()

    result = collator(make_training_batch_input())

    padding_positions = result.caption_attention_mask == 0
    token_positions = result.caption_attention_mask == 1

    assert torch.all(result.labels[padding_positions] == -100)
    assert torch.all(result.labels[token_positions] != -100)


def test_training_collator_uses_custom_label_pad_token_id() -> None:
    collator = make_training_collator(
        label_pad_token_id=-55,
    )

    result = collator(make_training_batch_input())

    padding_positions = result.caption_attention_mask == 0

    assert torch.all(result.labels[padding_positions] == -55)


def test_training_collator_supports_max_length_padding() -> None:
    collator = make_training_collator(
        max_length=8,
        padding="max_length",
    )

    result = collator(make_training_batch_input())

    assert result.labels.shape == (2, 8)
    assert result.caption_attention_mask.shape == (2, 8)


def test_training_collator_truncates_long_caption() -> None:
    collator = make_training_collator(
        max_length=4,
        padding="max_length",
        truncation=True,
    )

    result = collator(
        [
            (
                make_image(),
                "one two three four five six seven",
                "image.jpg",
            )
        ]
    )

    assert result.labels.shape == (1, 4)
    assert torch.all(result.caption_attention_mask == 1)


def test_training_collator_rejects_empty_batch() -> None:
    collator = make_training_collator()

    with pytest.raises(ValueError, match="cannot be empty"):
        collator([])


def test_training_collator_rejects_non_tuple_sample() -> None:
    collator = make_training_collator()

    with pytest.raises(TypeError, match="must be a tuple"):
        collator(
            [
                [
                    make_image(),
                    "caption",
                    "image.jpg",
                ]
            ]  # type: ignore[list-item]
        )


def test_training_collator_rejects_wrong_sample_length() -> None:
    collator = make_training_collator()

    with pytest.raises(ValueError, match="exactly three items"):
        collator(
            [
                (
                    make_image(),
                    "caption",
                )
            ]  # type: ignore[list-item]
        )


def test_training_collator_rejects_non_string_caption() -> None:
    collator = make_training_collator()

    with pytest.raises(TypeError, match="Caption"):
        collator(
            [
                (
                    make_image(),
                    123,
                    "image.jpg",
                )
            ]  # type: ignore[list-item]
        )


def test_training_collator_rejects_empty_caption() -> None:
    collator = make_training_collator()

    with pytest.raises(ValueError, match="Caption"):
        collator(
            [
                (
                    make_image(),
                    "   ",
                    "image.jpg",
                )
            ]
        )


def test_training_collator_rejects_non_string_image_name() -> None:
    collator = make_training_collator()

    with pytest.raises(TypeError, match="Image name"):
        collator(
            [
                (
                    make_image(),
                    "caption",
                    123,
                )
            ]  # type: ignore[list-item]
        )


def test_training_collator_rejects_empty_image_name() -> None:
    collator = make_training_collator()

    with pytest.raises(ValueError, match="Image name"):
        collator(
            [
                (
                    make_image(),
                    "caption",
                    "   ",
                )
            ]
        )


def test_training_collator_rejects_missing_input_ids() -> None:
    tokenizer = make_tokenizer()
    tokenizer.side_effect = None
    tokenizer.return_value = {
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    collator = make_training_collator(tokenizer=tokenizer)

    with pytest.raises(KeyError, match="input_ids"):
        collator(
            [
                (
                    make_image(),
                    "caption",
                    "image.jpg",
                )
            ]
        )


def test_training_collator_rejects_non_tensor_input_ids() -> None:
    tokenizer = make_tokenizer()
    tokenizer.side_effect = None
    tokenizer.return_value = {
        "input_ids": [[1, 2]],
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    collator = make_training_collator(tokenizer=tokenizer)

    with pytest.raises(TypeError, match="input_ids"):
        collator(
            [
                (
                    make_image(),
                    "caption",
                    "image.jpg",
                )
            ]
        )


def test_training_collator_rejects_non_2d_input_ids() -> None:
    tokenizer = make_tokenizer()
    tokenizer.side_effect = None
    tokenizer.return_value = {
        "input_ids": torch.ones(2, dtype=torch.long),
        "attention_mask": torch.ones(2, dtype=torch.long),
    }
    collator = make_training_collator(tokenizer=tokenizer)

    with pytest.raises(ValueError, match="2D Tensor"):
        collator(
            [
                (
                    make_image(),
                    "caption",
                    "image.jpg",
                )
            ]
        )


def test_training_collator_rejects_missing_attention_mask() -> None:
    tokenizer = make_tokenizer()
    tokenizer.side_effect = None
    tokenizer.return_value = {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
    }
    collator = make_training_collator(tokenizer=tokenizer)

    with pytest.raises(KeyError, match="attention_mask"):
        collator(
            [
                (
                    make_image(),
                    "caption",
                    "image.jpg",
                )
            ]
        )


def test_training_collator_rejects_non_tensor_attention_mask() -> None:
    tokenizer = make_tokenizer()
    tokenizer.side_effect = None
    tokenizer.return_value = {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "attention_mask": [[1, 1]],
    }
    collator = make_training_collator(tokenizer=tokenizer)

    with pytest.raises(TypeError, match="attention_mask"):
        collator(
            [
                (
                    make_image(),
                    "caption",
                    "image.jpg",
                )
            ]
        )


def test_training_collator_rejects_non_2d_attention_mask() -> None:
    tokenizer = make_tokenizer()
    tokenizer.side_effect = None
    tokenizer.return_value = {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(2, dtype=torch.long),
    }
    collator = make_training_collator(tokenizer=tokenizer)

    with pytest.raises(ValueError, match="2D Tensor"):
        collator(
            [
                (
                    make_image(),
                    "caption",
                    "image.jpg",
                )
            ]
        )


def test_training_collator_rejects_tokenizer_shape_mismatch() -> None:
    tokenizer = make_tokenizer()
    tokenizer.side_effect = None
    tokenizer.return_value = {
        "input_ids": torch.ones(1, 3, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    collator = make_training_collator(tokenizer=tokenizer)

    with pytest.raises(ValueError, match="same shape"):
        collator(
            [
                (
                    make_image(),
                    "caption",
                    "image.jpg",
                )
            ]
        )


def test_training_collator_rejects_unexpected_tokenizer_batch_size() -> None:
    tokenizer = make_tokenizer()
    tokenizer.side_effect = None
    tokenizer.return_value = {
        "input_ids": torch.ones(2, 3, dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
    }
    collator = make_training_collator(tokenizer=tokenizer)

    with pytest.raises(RuntimeError, match="unexpected batch size"):
        collator(
            [
                (
                    make_image(),
                    "caption",
                    "image.jpg",
                )
            ]
        )


def test_evaluation_collator_builds_expected_batch() -> None:
    collator = EvaluationCollator()
    source_batch = make_evaluation_batch_input()

    result = collator(source_batch)

    assert isinstance(result, EvaluationBatch)
    assert result.images.shape == (2, 3, 8, 8)
    assert result.reference_captions == [
        [
            "a dog runs",
            "a dog is running outside",
        ],
        [
            "two children play",
            "children are playing outside",
        ],
    ]
    assert result.image_names == [
        "dog.jpg",
        "children.jpg",
    ]


def test_evaluation_collator_copies_reference_lists() -> None:
    collator = EvaluationCollator()
    source_batch = make_evaluation_batch_input()

    source_references = source_batch[0][1]
    result = collator(source_batch)

    assert result.reference_captions[0] is not source_references

    result.reference_captions[0].append("new caption")

    assert "new caption" not in source_references


def test_evaluation_collator_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        EvaluationCollator()([])


def test_evaluation_collator_rejects_non_tuple_sample() -> None:
    with pytest.raises(TypeError, match="must be a tuple"):
        EvaluationCollator()(
            [
                [
                    make_image(),
                    ["caption"],
                    "image.jpg",
                ]
            ]  # type: ignore[list-item]
        )


def test_evaluation_collator_rejects_wrong_sample_length() -> None:
    with pytest.raises(ValueError, match="exactly three items"):
        EvaluationCollator()(
            [
                (
                    make_image(),
                    ["caption"],
                )
            ]  # type: ignore[list-item]
        )


def test_evaluation_collator_rejects_non_list_references() -> None:
    with pytest.raises(TypeError, match="must be a list"):
        EvaluationCollator()(
            [
                (
                    make_image(),
                    ("caption",),
                    "image.jpg",
                )
            ]  # type: ignore[list-item]
        )


def test_evaluation_collator_rejects_empty_references() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        EvaluationCollator()(
            [
                (
                    make_image(),
                    [],
                    "image.jpg",
                )
            ]
        )


def test_evaluation_collator_rejects_non_string_reference() -> None:
    with pytest.raises(TypeError, match="Reference caption"):
        EvaluationCollator()(
            [
                (
                    make_image(),
                    ["valid", 123],
                    "image.jpg",
                )
            ]  # type: ignore[list-item]
        )


def test_evaluation_collator_rejects_empty_reference() -> None:
    with pytest.raises(ValueError, match="Reference caption"):
        EvaluationCollator()(
            [
                (
                    make_image(),
                    ["valid", "   "],
                    "image.jpg",
                )
            ]
        )


def test_evaluation_collator_rejects_non_string_image_name() -> None:
    with pytest.raises(TypeError, match="Image name"):
        EvaluationCollator()(
            [
                (
                    make_image(),
                    ["caption"],
                    123,
                )
            ]  # type: ignore[list-item]
        )


def test_evaluation_collator_rejects_empty_image_name() -> None:
    with pytest.raises(ValueError, match="Image name"):
        EvaluationCollator()(
            [
                (
                    make_image(),
                    ["caption"],
                    "   ",
                )
            ]
        )