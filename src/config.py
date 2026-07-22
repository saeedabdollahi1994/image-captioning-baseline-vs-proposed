from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.data.dataloaders import DataLoaderSettings
from src.transforms import TransformSettings


DEFAULT_DATA_CONFIG_PATH = Path("configs/data.yaml")


class ConfigurationError(ValueError):
    """Raised when a configuration file is missing or invalid."""


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    """Resolved dataset paths and dataset metadata."""

    name: str
    images_dir: Path
    captions_path: Path
    split_path: Path

    def validate(
        self,
        *,
        check_paths: bool = True,
    ) -> None:
        """Validate dataset metadata and optionally check filesystem paths."""
        if not isinstance(self.name, str):
            raise TypeError(
                "dataset.name must be a string, "
                f"not {type(self.name).__name__}."
            )

        if not self.name.strip():
            raise ConfigurationError(
                "dataset.name cannot be empty."
            )

        for field_name, value in (
            ("images_dir", self.images_dir),
            ("captions_path", self.captions_path),
            ("split_path", self.split_path),
        ):
            if not isinstance(value, Path):
                raise TypeError(
                    f"dataset.{field_name} must be a pathlib.Path, "
                    f"not {type(value).__name__}."
                )

        if not check_paths:
            return

        if not self.images_dir.exists():
            raise FileNotFoundError(
                "Dataset image directory does not exist: "
                f"{self.images_dir}"
            )

        if not self.images_dir.is_dir():
            raise NotADirectoryError(
                "dataset.images_dir must point to a directory: "
                f"{self.images_dir}"
            )

        if not self.captions_path.exists():
            raise FileNotFoundError(
                "Caption file does not exist: "
                f"{self.captions_path}"
            )

        if not self.captions_path.is_file():
            raise ConfigurationError(
                "dataset.captions_path must point to a file: "
                f"{self.captions_path}"
            )

        if not self.split_path.exists():
            raise FileNotFoundError(
                "Dataset split file does not exist: "
                f"{self.split_path}"
            )

        if not self.split_path.is_file():
            raise ConfigurationError(
                "dataset.split_path must point to a file: "
                f"{self.split_path}"
            )


@dataclass(frozen=True, slots=True)
class DataPipelineConfig:
    """Complete configuration required by the data pipeline."""

    dataset: DatasetSettings
    transforms: TransformSettings
    dataloader: DataLoaderSettings
    config_path: Path
    project_root: Path

    def validate(
        self,
        *,
        check_paths: bool = True,
    ) -> None:
        """Validate all nested data-pipeline settings."""
        self.dataset.validate(
            check_paths=check_paths,
        )
        self.transforms.validate()
        self.dataloader.validate()

        if not isinstance(self.config_path, Path):
            raise TypeError(
                "config_path must be a pathlib.Path, "
                f"not {type(self.config_path).__name__}."
            )

        if not isinstance(self.project_root, Path):
            raise TypeError(
                "project_root must be a pathlib.Path, "
                f"not {type(self.project_root).__name__}."
            )


def _require_mapping(
    value: Any,
    *,
    location: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            f"{location} must be a YAML mapping."
        )

    return value


def _require_section(
    config: Mapping[str, Any],
    section_name: str,
) -> Mapping[str, Any]:
    if section_name not in config:
        raise ConfigurationError(
            f"Missing required configuration section: "
            f"{section_name}"
        )

    return _require_mapping(
        config[section_name],
        location=section_name,
    )


def _require_value(
    section: Mapping[str, Any],
    *,
    section_name: str,
    key: str,
) -> Any:
    if key not in section:
        raise ConfigurationError(
            f"Missing required configuration value: "
            f"{section_name}.{key}"
        )

    return section[key]


def _reject_unknown_keys(
    section: Mapping[str, Any],
    *,
    section_name: str,
    allowed_keys: set[str],
) -> None:
    unknown_keys = set(section) - allowed_keys

    if unknown_keys:
        formatted_keys = ", ".join(
            sorted(str(key) for key in unknown_keys)
        )
        raise ConfigurationError(
            f"Unknown key(s) in {section_name}: "
            f"{formatted_keys}"
        )


def _read_yaml_config(
    config_path: Path,
) -> Mapping[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    if not config_path.is_file():
        raise ConfigurationError(
            "Configuration path must point to a file: "
            f"{config_path}"
        )

    try:
        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            loaded_config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"Invalid YAML in {config_path}: {error}"
        ) from error

    if loaded_config is None:
        raise ConfigurationError(
            f"Configuration file is empty: {config_path}"
        )

    return _require_mapping(
        loaded_config,
        location="configuration root",
    )


def _resolve_project_root(
    *,
    config_path: Path,
    project_root: str | Path | None,
) -> Path:
    if project_root is not None:
        return Path(project_root).expanduser().resolve()

    resolved_config_path = config_path.resolve()

    if resolved_config_path.parent.name == "configs":
        return resolved_config_path.parent.parent

    return Path.cwd().resolve()


def _resolve_path(
    raw_path: Any,
    *,
    field_name: str,
    project_root: Path,
) -> Path:
    if not isinstance(raw_path, (str, Path)):
        raise TypeError(
            f"{field_name} must be a string or pathlib.Path, "
            f"not {type(raw_path).__name__}."
        )

    path = Path(raw_path).expanduser()

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def _as_three_float_tuple(
    value: Any,
    *,
    field_name: str,
) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"{field_name} must be a list or tuple."
        )

    if len(value) != 3:
        raise ConfigurationError(
            f"{field_name} must contain exactly three values."
        )

    return (
        value[0],
        value[1],
        value[2],
    )


def _as_two_float_tuple(
    value: Any,
    *,
    field_name: str,
) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"{field_name} must be a list or tuple."
        )

    if len(value) != 2:
        raise ConfigurationError(
            f"{field_name} must contain exactly two values."
        )

    return (
        value[0],
        value[1],
    )


def _build_dataset_settings(
    section: Mapping[str, Any],
    *,
    project_root: Path,
) -> DatasetSettings:
    allowed_keys = {
        "name",
        "images_dir",
        "captions_path",
        "split_path",
    }
    _reject_unknown_keys(
        section,
        section_name="dataset",
        allowed_keys=allowed_keys,
    )

    name = _require_value(
        section,
        section_name="dataset",
        key="name",
    )

    if not isinstance(name, str):
        raise TypeError(
            "dataset.name must be a string, "
            f"not {type(name).__name__}."
        )

    return DatasetSettings(
        name=name.strip(),
        images_dir=_resolve_path(
            _require_value(
                section,
                section_name="dataset",
                key="images_dir",
            ),
            field_name="dataset.images_dir",
            project_root=project_root,
        ),
        captions_path=_resolve_path(
            _require_value(
                section,
                section_name="dataset",
                key="captions_path",
            ),
            field_name="dataset.captions_path",
            project_root=project_root,
        ),
        split_path=_resolve_path(
            _require_value(
                section,
                section_name="dataset",
                key="split_path",
            ),
            field_name="dataset.split_path",
            project_root=project_root,
        ),
    )


def _build_transform_settings(
    section: Mapping[str, Any],
) -> TransformSettings:
    allowed_keys = {
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
    }
    _reject_unknown_keys(
        section,
        section_name="transforms",
        allowed_keys=allowed_keys,
    )

    settings = TransformSettings(
        image_size=_require_value(
            section,
            section_name="transforms",
            key="image_size",
        ),
        resize_size=_require_value(
            section,
            section_name="transforms",
            key="resize_size",
        ),
        image_mean=_as_three_float_tuple(
            _require_value(
                section,
                section_name="transforms",
                key="image_mean",
            ),
            field_name="transforms.image_mean",
        ),
        image_std=_as_three_float_tuple(
            _require_value(
                section,
                section_name="transforms",
                key="image_std",
            ),
            field_name="transforms.image_std",
        ),
        train_crop_scale=_as_two_float_tuple(
            _require_value(
                section,
                section_name="transforms",
                key="train_crop_scale",
            ),
            field_name="transforms.train_crop_scale",
        ),
        train_crop_ratio=_as_two_float_tuple(
            _require_value(
                section,
                section_name="transforms",
                key="train_crop_ratio",
            ),
            field_name="transforms.train_crop_ratio",
        ),
        horizontal_flip_probability=_require_value(
            section,
            section_name="transforms",
            key="horizontal_flip_probability",
        ),
        brightness_jitter=_require_value(
            section,
            section_name="transforms",
            key="brightness_jitter",
        ),
        contrast_jitter=_require_value(
            section,
            section_name="transforms",
            key="contrast_jitter",
        ),
        saturation_jitter=_require_value(
            section,
            section_name="transforms",
            key="saturation_jitter",
        ),
        hue_jitter=_require_value(
            section,
            section_name="transforms",
            key="hue_jitter",
        ),
    )
    settings.validate()
    return settings


def _build_dataloader_settings(
    dataloader_section: Mapping[str, Any],
    tokenization_section: Mapping[str, Any],
) -> DataLoaderSettings:
    dataloader_allowed_keys = {
        "train_batch_size",
        "evaluation_batch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "shuffle_train",
        "drop_last_train",
        "seed",
    }
    tokenization_allowed_keys = {
        "max_caption_length",
        "padding",
        "truncation",
        "label_pad_token_id",
    }

    _reject_unknown_keys(
        dataloader_section,
        section_name="dataloader",
        allowed_keys=dataloader_allowed_keys,
    )
    _reject_unknown_keys(
        tokenization_section,
        section_name="tokenization",
        allowed_keys=tokenization_allowed_keys,
    )

    settings = DataLoaderSettings(
        train_batch_size=_require_value(
            dataloader_section,
            section_name="dataloader",
            key="train_batch_size",
        ),
        evaluation_batch_size=_require_value(
            dataloader_section,
            section_name="dataloader",
            key="evaluation_batch_size",
        ),
        num_workers=_require_value(
            dataloader_section,
            section_name="dataloader",
            key="num_workers",
        ),
        pin_memory=_require_value(
            dataloader_section,
            section_name="dataloader",
            key="pin_memory",
        ),
        persistent_workers=_require_value(
            dataloader_section,
            section_name="dataloader",
            key="persistent_workers",
        ),
        prefetch_factor=_require_value(
            dataloader_section,
            section_name="dataloader",
            key="prefetch_factor",
        ),
        shuffle_train=_require_value(
            dataloader_section,
            section_name="dataloader",
            key="shuffle_train",
        ),
        drop_last_train=_require_value(
            dataloader_section,
            section_name="dataloader",
            key="drop_last_train",
        ),
        seed=_require_value(
            dataloader_section,
            section_name="dataloader",
            key="seed",
        ),
        max_caption_length=_require_value(
            tokenization_section,
            section_name="tokenization",
            key="max_caption_length",
        ),
        padding=_require_value(
            tokenization_section,
            section_name="tokenization",
            key="padding",
        ),
        truncation=_require_value(
            tokenization_section,
            section_name="tokenization",
            key="truncation",
        ),
        label_pad_token_id=_require_value(
            tokenization_section,
            section_name="tokenization",
            key="label_pad_token_id",
        ),
    )
    settings.validate()
    return settings


def load_data_config(
    config_path: str | Path = DEFAULT_DATA_CONFIG_PATH,
    *,
    project_root: str | Path | None = None,
    check_paths: bool = True,
) -> DataPipelineConfig:
    """Load, resolve, and validate the complete data configuration."""
    resolved_config_path = Path(
        config_path
    ).expanduser().resolve()

    raw_config = _read_yaml_config(
        resolved_config_path
    )

    allowed_root_sections = {
        "dataset",
        "transforms",
        "dataloader",
        "tokenization",
    }
    _reject_unknown_keys(
        raw_config,
        section_name="configuration root",
        allowed_keys=allowed_root_sections,
    )

    resolved_project_root = _resolve_project_root(
        config_path=resolved_config_path,
        project_root=project_root,
    )

    dataset_section = _require_section(
        raw_config,
        "dataset",
    )
    transforms_section = _require_section(
        raw_config,
        "transforms",
    )
    dataloader_section = _require_section(
        raw_config,
        "dataloader",
    )
    tokenization_section = _require_section(
        raw_config,
        "tokenization",
    )

    config = DataPipelineConfig(
        dataset=_build_dataset_settings(
            dataset_section,
            project_root=resolved_project_root,
        ),
        transforms=_build_transform_settings(
            transforms_section,
        ),
        dataloader=_build_dataloader_settings(
            dataloader_section,
            tokenization_section,
        ),
        config_path=resolved_config_path,
        project_root=resolved_project_root,
    )
    config.validate(
        check_paths=check_paths,
    )
    return config