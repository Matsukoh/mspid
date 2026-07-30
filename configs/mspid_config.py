from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


class ModelConfig(ConfigBase):
    h_dims: list[int] = [256, 256]
    time_dim: int = 16
    L: int = 10
    T: int = 2
    w: float = 500
    q_grad_norm: bool = True


class TrainConfig(ConfigBase):
    random_seed: int = 17
    lr: float = 1e-4
    preliminary_exp_n: int = 2000
    total_it: int = 1000000
    batch_size: int = 100
    buffer_capacity: int = 1000000
    polyak: float = 0.995
    training_alg: str = "MSPID"


class LogConfig(ConfigBase):
    wandb_project: str = "MSPID"
    # wandb_mode: str = "offline"
    wandb_mode: str = "online"
    wandb: bool = True
    save_model: bool = False


class TotalConfig(BaseSettings):
    model_config = SettingsConfigDict(
        extra="forbid",
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
        env_nested_delimiter="__",
        env_prefix="EXPERIMENT_",
        env_file=".env",
        env_file_encoding="utf-8",
    )
    experiment_name: str = "test"

    model: ModelConfig = Field(default_factory=ModelConfig)

    train: TrainConfig = Field(default_factory=TrainConfig)

    log: LogConfig = Field(default_factory=LogConfig)
