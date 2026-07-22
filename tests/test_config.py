"""Tests for loading and validating data-pipeline configuration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.config import (
    DEFAULT_DATA_CONFIG_PATH,
    ConfigurationError,
    DataPipelineConfig,
    DatasetSettings,
    load_data_config,
)
from src.data.dataloaders import DataLoaderSettings
from src.transforms import TransformSettings


def _valid_config_dict() -> dict[str, Any]:
    return {
        "dataset": {
            "name": "flickr8k",
            "images_dir": "data/raw/images",
            "captions_path": "data/raw/captions.txt",
            "split_path": "data/splits/shared_split.json",
        },
        "transforms": {
            "image_size": 224,
            "resize_size": 256,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "train_crop_scale": [0.85, 1.0],
            "train_crop_ratio": [0.9, 1.1],
            "horizontal_flip_probability": 0.5,
            "brightness_jitter": 0.1,
            "contrast_jitter": 0.1,
            "saturation_jitter": 0.1,
            "hue_jitter": 0.02,
        },
        "dataloader": {
            "train_batch_size": 16,
            "evaluation_batch_size": 16,
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "shuffle_train": True,
            "drop_last_train": False,
            "seed": 42,
        },
        "tokenization": {
            "max_caption_length": 40,
            "padding": "longest",
            "truncation": True,
            "label_pad_token_id": -100,
        },
    }


def _create_project_layout(
    tmp_path: Path,
) -> dict[str, Path]:
    project_root = tmp_path / "project"
    config_dir = project_root / "configs"
    images_dir = project_root / "data" / "raw" / "images"
    captions_path = (
        project_root / "data" / "raw" / "captions.txt"
    )
    split_path = (
        project_root / "data" / "splits" / "shared_split.json"
    )
    config_path = config_dir / "data.yaml"

    config_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    captions_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    captions_path.write_text(
        "image,caption\n",
        encoding="utf-8",
    )
    split_path.write_text(
        "{}",
        encoding="utf-8",
    )

    return {
        "project_root": project_root,
        "config_path": config_path,
        "images_dir": images_dir,
        "captions_path": captions_path,
        "split_path": split_path,
    }


def _write_yaml(
    path: Path,
    content: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            content,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _load_valid_config(
    tmp_path: Path,
    *,
    check_paths: bool = True,
) -> tuple[DataPipelineConfig, dict[str, Path]]:
    paths = _create_project_layout(tmp_path)
    _write_yaml(
        paths["config_path"],
        _valid_config_dict(),
    )
    config = load_data_config(
        paths["config_path"],
        check_paths=check_paths,
    )
    return config, paths


def _valid_dataset_settings(
    tmp_path: Path,
) -> DatasetSettings:
    images_dir = tmp_path / "images"
    captions_path = tmp_path / "captions.txt"
    split_path = tmp_path / "split.json"

    images_dir.mkdir()
    captions_path.write_text(
        "image,caption\n",
        encoding="utf-8",
    )
    split_path.write_text(
        "{}",
        encoding="utf-8",
    )

    return DatasetSettings(
        name="flickr8k",
        images_dir=images_dir,
        captions_path=captions_path,
        split_path=split_path,
    )


def _valid_pipeline_config(
    tmp_path: Path,
) -> DataPipelineConfig:
    return DataPipelineConfig(
        dataset=_valid_dataset_settings(tmp_path),
        transforms=TransformSettings(),
        dataloader=DataLoaderSettings(),
        config_path=tmp_path / "configs" / "data.yaml",
        project_root=tmp_path,
    )


def test_default_data_config_path() -> None:
    assert DEFAULT_DATA_CONFIG_PATH == Path(
        "configs/data.yaml"
    )


def test_configuration_error_is_value_error() -> None:
    assert issubclass(
        ConfigurationError,
        ValueError,
    )


def test_dataset_settings_are_frozen(
    tmp_path: Path,
) -> None:
    settings = _valid_dataset_settings(tmp_path)

    with pytest.raises(FrozenInstanceError):
        settings.name = "other"  # type: ignore[misc]


def test_data_pipeline_config_is_frozen(
    tmp_path: Path,
) -> None:
    config = _valid_pipeline_config(tmp_path)

    with pytest.raises(FrozenInstanceError):
        config.project_root = Path(".")  # type: ignore[misc]


def test_dataset_settings_validate_existing_paths(
    tmp_path: Path,
) -> None:
    _valid_dataset_settings(tmp_path).validate()


def test_dataset_settings_can_skip_path_checks(
    tmp_path: Path,
) -> None:
    settings = DatasetSettings(
        name="flickr8k",
        images_dir=tmp_path / "missing-images",
        captions_path=tmp_path / "missing-captions.txt",
        split_path=tmp_path / "missing-split.json",
    )

    settings.validate(check_paths=False)


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "\t",
        "\n",
    ],
)
def test_dataset_name_cannot_be_empty(
    tmp_path: Path,
    name: str,
) -> None:
    settings = _valid_dataset_settings(tmp_path)
    invalid = DatasetSettings(
        name=name,
        images_dir=settings.images_dir,
        captions_path=settings.captions_path,
        split_path=settings.split_path,
    )

    with pytest.raises(
        ConfigurationError,
        match="dataset.name cannot be empty",
    ):
        invalid.validate()


@pytest.mark.parametrize(
    "name",
    [
        None,
        123,
        True,
        [],
    ],
)
def test_dataset_name_must_be_string(
    tmp_path: Path,
    name: object,
) -> None:
    settings = _valid_dataset_settings(tmp_path)
    invalid = DatasetSettings(
        name=name,  # type: ignore[arg-type]
        images_dir=settings.images_dir,
        captions_path=settings.captions_path,
        split_path=settings.split_path,
    )

    with pytest.raises(
        TypeError,
        match="dataset.name must be a string",
    ):
        invalid.validate(check_paths=False)


@pytest.mark.parametrize(
    "field_name",
    [
        "images_dir",
        "captions_path",
        "split_path",
    ],
)
def test_dataset_paths_must_be_path_objects(
    tmp_path: Path,
    field_name: str,
) -> None:
    settings = _valid_dataset_settings(tmp_path)
    values = {
        "name": settings.name,
        "images_dir": settings.images_dir,
        "captions_path": settings.captions_path,
        "split_path": settings.split_path,
    }
    values[field_name] = "not-a-path"

    invalid = DatasetSettings(**values)  # type: ignore[arg-type]

    with pytest.raises(
        TypeError,
        match=rf"dataset\.{field_name} must be a pathlib\.Path",
    ):
        invalid.validate(check_paths=False)


def test_missing_images_directory_is_rejected(
    tmp_path: Path,
) -> None:
    settings = _valid_dataset_settings(tmp_path)
    settings.images_dir.rmdir()

    with pytest.raises(
        FileNotFoundError,
        match="Dataset image directory does not exist",
    ):
        settings.validate()


def test_images_path_must_be_directory(
    tmp_path: Path,
) -> None:
    settings = _valid_dataset_settings(tmp_path)
    settings.images_dir.rmdir()
    settings.images_dir.write_text(
        "not a directory",
        encoding="utf-8",
    )

    with pytest.raises(
        NotADirectoryError,
        match="images_dir must point to a directory",
    ):
        settings.validate()


def test_missing_captions_file_is_rejected(
    tmp_path: Path,
) -> None:
    settings = _valid_dataset_settings(tmp_path)
    settings.captions_path.unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Caption file does not exist",
    ):
        settings.validate()


def test_captions_path_must_be_file(
    tmp_path: Path,
) -> None:
    settings = _valid_dataset_settings(tmp_path)
    settings.captions_path.unlink()
    settings.captions_path.mkdir()

    with pytest.raises(
        ConfigurationError,
        match="captions_path must point to a file",
    ):
        settings.validate()


def test_missing_split_file_is_rejected(
    tmp_path: Path,
) -> None:
    settings = _valid_dataset_settings(tmp_path)
    settings.split_path.unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Dataset split file does not exist",
    ):
        settings.validate()


def test_split_path_must_be_file(
    tmp_path: Path,
) -> None:
    settings = _valid_dataset_settings(tmp_path)
    settings.split_path.unlink()
    settings.split_path.mkdir()

    with pytest.raises(
        ConfigurationError,
        match="split_path must point to a file",
    ):
        settings.validate()


def test_data_pipeline_config_validates_nested_settings(
    tmp_path: Path,
) -> None:
    _valid_pipeline_config(tmp_path).validate()


def test_data_pipeline_config_can_skip_dataset_path_checks(
    tmp_path: Path,
) -> None:
    config = DataPipelineConfig(
        dataset=DatasetSettings(
            name="flickr8k",
            images_dir=tmp_path / "missing-images",
            captions_path=tmp_path / "missing-captions.txt",
            split_path=tmp_path / "missing-split.json",
        ),
        transforms=TransformSettings(),
        dataloader=DataLoaderSettings(),
        config_path=tmp_path / "configs" / "data.yaml",
        project_root=tmp_path,
    )

    config.validate(check_paths=False)


def test_data_pipeline_config_requires_path_config_path(
    tmp_path: Path,
) -> None:
    config = _valid_pipeline_config(tmp_path)
    invalid = DataPipelineConfig(
        dataset=config.dataset,
        transforms=config.transforms,
        dataloader=config.dataloader,
        config_path="configs/data.yaml",  # type: ignore[arg-type]
        project_root=config.project_root,
    )

    with pytest.raises(
        TypeError,
        match="config_path must be a pathlib.Path",
    ):
        invalid.validate()


def test_data_pipeline_config_requires_path_project_root(
    tmp_path: Path,
) -> None:
    config = _valid_pipeline_config(tmp_path)
    invalid = DataPipelineConfig(
        dataset=config.dataset,
        transforms=config.transforms,
        dataloader=config.dataloader,
        config_path=config.config_path,
        project_root="project",  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError,
        match="project_root must be a pathlib.Path",
    ):
        invalid.validate()


def test_load_data_config_returns_expected_types(
    tmp_path: Path,
) -> None:
    config, _ = _load_valid_config(tmp_path)

    assert isinstance(config, DataPipelineConfig)
    assert isinstance(config.dataset, DatasetSettings)
    assert isinstance(config.transforms, TransformSettings)
    assert isinstance(config.dataloader, DataLoaderSettings)
    assert isinstance(config.config_path, Path)
    assert isinstance(config.project_root, Path)


def test_load_data_config_reads_expected_values(
    tmp_path: Path,
) -> None:
    config, _ = _load_valid_config(tmp_path)

    assert config.dataset.name == "flickr8k"
    assert config.transforms.image_size == 224
    assert config.transforms.resize_size == 256
    assert config.transforms.image_mean == (
        0.5,
        0.5,
        0.5,
    )
    assert config.transforms.image_std == (
        0.5,
        0.5,
        0.5,
    )
    assert config.transforms.train_crop_scale == (
        0.85,
        1.0,
    )
    assert config.transforms.train_crop_ratio == (
        0.9,
        1.1,
    )
    assert config.dataloader.train_batch_size == 16
    assert config.dataloader.evaluation_batch_size == 16
    assert config.dataloader.num_workers == 2
    assert config.dataloader.pin_memory is True
    assert config.dataloader.persistent_workers is True
    assert config.dataloader.prefetch_factor == 2
    assert config.dataloader.seed == 42
    assert config.dataloader.max_caption_length == 40
    assert config.dataloader.padding == "longest"
    assert config.dataloader.truncation is True
    assert config.dataloader.label_pad_token_id == -100


def test_relative_paths_are_resolved_from_project_root(
    tmp_path: Path,
) -> None:
    config, paths = _load_valid_config(tmp_path)

    assert config.dataset.images_dir == paths[
        "images_dir"
    ].resolve()
    assert config.dataset.captions_path == paths[
        "captions_path"
    ].resolve()
    assert config.dataset.split_path == paths[
        "split_path"
    ].resolve()


def test_config_inside_configs_infers_project_root(
    tmp_path: Path,
) -> None:
    config, paths = _load_valid_config(tmp_path)

    assert config.project_root == paths[
        "project_root"
    ].resolve()


def test_config_path_is_resolved_to_absolute_path(
    tmp_path: Path,
) -> None:
    config, paths = _load_valid_config(tmp_path)

    assert config.config_path == paths[
        "config_path"
    ].resolve()
    assert config.config_path.is_absolute()


def test_explicit_project_root_overrides_inferred_root(
    tmp_path: Path,
) -> None:
    paths = _create_project_layout(tmp_path)
    alternate_root = tmp_path / "alternate-project"
    alternate_images = alternate_root / "data" / "raw" / "images"
    alternate_captions = (
        alternate_root / "data" / "raw" / "captions.txt"
    )
    alternate_split = (
        alternate_root / "data" / "splits" / "shared_split.json"
    )

    alternate_images.mkdir(parents=True)
    alternate_captions.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    alternate_split.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    alternate_captions.write_text(
        "image,caption\n",
        encoding="utf-8",
    )
    alternate_split.write_text(
        "{}",
        encoding="utf-8",
    )

    _write_yaml(
        paths["config_path"],
        _valid_config_dict(),
    )

    config = load_data_config(
        paths["config_path"],
        project_root=alternate_root,
    )

    assert config.project_root == alternate_root.resolve()
    assert config.dataset.images_dir == (
        alternate_images.resolve()
    )
    assert config.dataset.captions_path == (
        alternate_captions.resolve()
    )
    assert config.dataset.split_path == (
        alternate_split.resolve()
    )


def test_absolute_dataset_paths_remain_absolute(
    tmp_path: Path,
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    raw_config["dataset"]["images_dir"] = str(
        paths["images_dir"].resolve()
    )
    raw_config["dataset"]["captions_path"] = str(
        paths["captions_path"].resolve()
    )
    raw_config["dataset"]["split_path"] = str(
        paths["split_path"].resolve()
    )
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    config = load_data_config(
        paths["config_path"]
    )

    assert config.dataset.images_dir == paths[
        "images_dir"
    ].resolve()
    assert config.dataset.captions_path == paths[
        "captions_path"
    ].resolve()
    assert config.dataset.split_path == paths[
        "split_path"
    ].resolve()


def test_dataset_name_is_trimmed(
    tmp_path: Path,
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    raw_config["dataset"]["name"] = "  flickr8k  "
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    config = load_data_config(
        paths["config_path"]
    )

    assert config.dataset.name == "flickr8k"


def test_missing_dataset_paths_can_be_skipped(
    tmp_path: Path,
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    raw_config["dataset"]["images_dir"] = (
        "missing/images"
    )
    raw_config["dataset"]["captions_path"] = (
        "missing/captions.txt"
    )
    raw_config["dataset"]["split_path"] = (
        "missing/split.json"
    )
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    config = load_data_config(
        paths["config_path"],
        check_paths=False,
    )

    assert not config.dataset.images_dir.exists()
    assert not config.dataset.captions_path.exists()
    assert not config.dataset.split_path.exists()


def test_missing_configuration_file_is_rejected(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(
        FileNotFoundError,
        match="Configuration file does not exist",
    ):
        load_data_config(missing)


def test_configuration_path_must_be_file(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config-directory"
    config_dir.mkdir()

    with pytest.raises(
        ConfigurationError,
        match="Configuration path must point to a file",
    ):
        load_data_config(config_dir)


def test_empty_configuration_file_is_rejected(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="Configuration file is empty",
    ):
        load_data_config(config_path)


def test_invalid_yaml_is_rejected(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "dataset: [\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError,
        match="Invalid YAML",
    ):
        load_data_config(config_path)


@pytest.mark.parametrize(
    "root_value",
    [
        [],
        "text",
        42,
        True,
    ],
)
def test_configuration_root_must_be_mapping(
    tmp_path: Path,
    root_value: object,
) -> None:
    config_path = tmp_path / "data.yaml"
    _write_yaml(
        config_path,
        root_value,
    )

    with pytest.raises(
        ConfigurationError,
        match="configuration root must be a YAML mapping",
    ):
        load_data_config(config_path)


@pytest.mark.parametrize(
    "section_name",
    [
        "dataset",
        "transforms",
        "dataloader",
        "tokenization",
    ],
)
def test_required_sections_must_exist(
    tmp_path: Path,
    section_name: str,
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    del raw_config[section_name]
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    with pytest.raises(
        ConfigurationError,
        match=rf"Missing required configuration section: {section_name}",
    ):
        load_data_config(
            paths["config_path"],
            check_paths=False,
        )


@pytest.mark.parametrize(
    "section_name",
    [
        "dataset",
        "transforms",
        "dataloader",
        "tokenization",
    ],
)
@pytest.mark.parametrize(
    "section_value",
    [
        [],
        "text",
        123,
    ],
)
def test_sections_must_be_mappings(
    tmp_path: Path,
    section_name: str,
    section_value: object,
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    raw_config[section_name] = section_value
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    with pytest.raises(
        ConfigurationError,
        match=rf"{section_name} must be a YAML mapping",
    ):
        load_data_config(
            paths["config_path"],
            check_paths=False,
        )


def test_unknown_root_key_is_rejected(
    tmp_path: Path,
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    raw_config["unknown"] = {}
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    with pytest.raises(
        ConfigurationError,
        match=r"Unknown key\(s\) in configuration root: unknown",
    ):
        load_data_config(
            paths["config_path"],
            check_paths=False,
        )


@pytest.mark.parametrize(
    ("section_name", "unknown_key"),
    [
        ("dataset", "root_dir"),
        ("transforms", "random_rotation"),
        ("dataloader", "timeout"),
        ("tokenization", "add_special_tokens"),
    ],
)
def test_unknown_section_keys_are_rejected(
    tmp_path: Path,
    section_name: str,
    unknown_key: str,
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    raw_config[section_name][unknown_key] = "value"
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    with pytest.raises(
        ConfigurationError,
        match=rf"Unknown key\(s\) in {section_name}: {unknown_key}",
    ):
        load_data_config(
            paths["config_path"],
            check_paths=False,
        )


@pytest.mark.parametrize(
    "key",
    [
        "name",
        "images_dir",
        "captions_path",
        "split_path",
    ],
)
def test_missing_dataset_values_are_rejected(
    tmp_path: Path,
    key: str,
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    del raw_config["dataset"][key]
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    with pytest.raises(
        ConfigurationError,
        match=rf"Missing required configuration value: dataset\.{key}",
    ):
        load_data_config(
            paths["config_path"],
            check_paths=False,
        )


@pytest.mark.parametrize(
    "key",
    [
        "image_size",
        "resize_size",
        "image_mean",
        "image_std",
        "train_crop_scale",
        "train_crop_ratio",
        "horizontal_flip_probability",
        "brightness_jitter",
        "contrast_jitter",
        "saturation_jitter",
        "hue_jitter",
    ],
)
def test_missing_transform_values_are_rejected(
    tmp_path: Path,
    key: str,
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    del raw_config["transforms"][key]
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    with pytest.raises(
        ConfigurationError,
        match=rf"Missing required configuration value: transforms\.{key}",
    ):
        load_data_config(
            paths["config_path"],
            check_paths=False,
        )


@pytest.mark.parametrize(
    "key",
    [
        "train_batch_size",
        "evaluation_batch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "shuffle_train",
        "drop_last_train",
        "seed",
    ],
)
def test_missing_dataloader_values_are_rejected(
    tmp_path: Path,
    key: str,
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    del raw_config["dataloader"][key]
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    with pytest.raises(
        ConfigurationError,
        match=rf"Missing required configuration value: dataloader\.{key}",
    ):
        load_data_config(
            paths["config_path"],
            check_paths=False,
        )


@pytest.mark.parametrize(
    "key",
    [
        "max_caption_length",
        "padding",
        "truncation",
        "label_pad_token_id",
    ],
)
def test_missing_tokenization_values_are_rejected(
    tmp_path: Path,
    key: str,
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    del raw_config["tokenization"][key]
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    with pytest.raises(
        ConfigurationError,
        match=rf"Missing required configuration value: tokenization\.{key}",
    ):
        load_data_config(
            paths["config_path"],
            check_paths=False,
        )


@pytest.mark.parametrize(
    "name",
    [
        1,
        True,
        [],
        None,
    ],
)
def test_loaded_dataset_name_must_be_string(
    tmp_path: Path,
    name: object,
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    raw_config["dataset"]["name"] = name
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    with pytest.raises(
        TypeError,
        match="dataset.name must be a string",
    ):
        load_data_config(
            paths["config_path"],
            check_paths=False,
        )


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "\t",
    ],
)
def test_loaded_dataset_name_cannot_be_blank(
    tmp_path: Path,
    name: str,
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    raw_config["dataset"]["name"] = name
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    with pytest.raises(
        ConfigurationError,
        match="dataset.name cannot be empty",
    ):
        load_data_config(
            paths["config_path"],
            check_paths=False,
        )


@pytest.mark.parametrize(
    "key",
    [
        "images_dir",
        "captions_path",
        "split_path",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        123,
        True,
        [],
        {},
    ],
)
def test_loaded_dataset_paths_must_be_strings_or_paths(
    tmp_path: Path,
    key: str,
    value: object,
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    raw_config["dataset"][key] = value
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    with pytest.raises(
        TypeError,
        match=rf"dataset\.{key} must be a string or pathlib\.Path",
    ):
        load_data_config(
            paths["config_path"],
            check_paths=False,
        )


@pytest.mark.parametrize(
    "key",
    [
        "image_mean",
        "image_std",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        "0.5,0.5,0.5",
        0.5,
        None,
        {},
    ],
)
def test_channel_settings_must_be_sequences(
    tmp_path: Path,
    key: str,
    value: object,
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    raw_config["transforms"][key] = value
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    with pytest.raises(
        TypeError,
        match=rf"transforms\.{key} must be a list or tuple",
    ):
        load_data_config(
            paths["config_path"],
            check_paths=False,
        )


@pytest.mark.parametrize(
    "key",
    [
        "image_mean",
        "image_std",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        [],
        [0.5],
        [0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
    ],
)
def test_channel_settings_must_have_three_values(
    tmp_path: Path,
    key: str,
    value: list[float],
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    raw_config["transforms"][key] = value
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    with pytest.raises(
        ConfigurationError,
        match=rf"transforms\.{key} must contain exactly three values",
    ):
        load_data_config(
            paths["config_path"],
            check_paths=False,
        )


@pytest.mark.parametrize(
    "key",
    [
        "train_crop_scale",
        "train_crop_ratio",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        "0.8,1.0",
        0.8,
        None,
        {},
    ],
)
def test_crop_settings_must_be_sequences(
    tmp_path: Path,
    key: str,
    value: object,
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    raw_config["transforms"][key] = value
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    with pytest.raises(
        TypeError,
        match=rf"transforms\.{key} must be a list or tuple",
    ):
        load_data_config(
            paths["config_path"],
            check_paths=False,
        )


@pytest.mark.parametrize(
    "key",
    [
        "train_crop_scale",
        "train_crop_ratio",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        [],
        [0.8],
        [0.8, 1.0, 1.1],
    ],
)
def test_crop_settings_must_have_two_values(
    tmp_path: Path,
    key: str,
    value: list[float],
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    raw_config["transforms"][key] = value
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    with pytest.raises(
        ConfigurationError,
        match=rf"transforms\.{key} must contain exactly two values",
    ):
        load_data_config(
            paths["config_path"],
            check_paths=False,
        )


def test_yaml_lists_are_converted_to_tuples(
    tmp_path: Path,
) -> None:
    config, _ = _load_valid_config(tmp_path)

    assert isinstance(
        config.transforms.image_mean,
        tuple,
    )
    assert isinstance(
        config.transforms.image_std,
        tuple,
    )
    assert isinstance(
        config.transforms.train_crop_scale,
        tuple,
    )
    assert isinstance(
        config.transforms.train_crop_ratio,
        tuple,
    )


@pytest.mark.parametrize(
    ("section", "key", "value", "error_type"),
    [
        ("transforms", "image_size", 0, ValueError),
        ("transforms", "resize_size", 100, ValueError),
        (
            "transforms",
            "horizontal_flip_probability",
            1.5,
            ValueError,
        ),
        ("transforms", "hue_jitter", 0.8, ValueError),
        ("dataloader", "train_batch_size", 0, ValueError),
        ("dataloader", "num_workers", -1, ValueError),
        (
            "dataloader",
            "persistent_workers",
            True,
            None,
        ),
        (
            "tokenization",
            "max_caption_length",
            0,
            ValueError,
        ),
        (
            "tokenization",
            "padding",
            "invalid",
            ValueError,
        ),
    ],
)
def test_nested_setting_validation_errors_propagate(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
    error_type: type[Exception] | None,
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    raw_config[section][key] = value

    if (
        section == "dataloader"
        and key == "persistent_workers"
        and value is True
    ):
        raw_config["dataloader"]["num_workers"] = 0
        expected_error = ValueError
    else:
        expected_error = error_type

    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    assert expected_error is not None

    with pytest.raises(expected_error):
        load_data_config(
            paths["config_path"],
            check_paths=False,
        )


def test_multiple_unknown_keys_are_sorted_in_error(
    tmp_path: Path,
) -> None:
    paths = _create_project_layout(tmp_path)
    raw_config = _valid_config_dict()
    raw_config["dataset"]["z_key"] = 1
    raw_config["dataset"]["a_key"] = 2
    _write_yaml(
        paths["config_path"],
        raw_config,
    )

    with pytest.raises(
        ConfigurationError,
        match=r"Unknown key\(s\) in dataset: a_key, z_key",
    ):
        load_data_config(
            paths["config_path"],
            check_paths=False,
        )


def test_config_outside_configs_uses_current_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_directory = tmp_path / "working"
    working_directory.mkdir()
    images_dir = working_directory / "data" / "raw" / "images"
    captions_path = (
        working_directory / "data" / "raw" / "captions.txt"
    )
    split_path = (
        working_directory / "data" / "splits" / "shared_split.json"
    )
    images_dir.mkdir(parents=True)
    captions_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    captions_path.write_text(
        "image,caption\n",
        encoding="utf-8",
    )
    split_path.write_text(
        "{}",
        encoding="utf-8",
    )

    config_path = tmp_path / "standalone.yaml"
    _write_yaml(
        config_path,
        _valid_config_dict(),
    )

    monkeypatch.chdir(working_directory)
    config = load_data_config(config_path)

    assert config.project_root == working_directory.resolve()
    assert config.dataset.images_dir == images_dir.resolve()
    assert config.dataset.captions_path == captions_path.resolve()
    assert config.dataset.split_path == split_path.resolve()