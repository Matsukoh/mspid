from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


class EnvironmentConfig(ConfigBase):
    time_limit: float = 30
    time_step: float = 0.25
    val_size: int = 100
    test_size: int = 500
    train_size: int = np.iinfo(np.uint32).max - 2000
    randomize_attributes: bool = False
    robot_sensor_range: float = 5


class RewardConfig(ConfigBase):
    success_reward: float = 1
    collision_penalty: float = -0.25
    discomfort_dist: float = 0.2
    discomfort_penalty_factor: float = 0.5


class SimulationConfig(ConfigBase):
    train_val_scenario: str = "circle_crossing"
    test_scenario: str = "circle_crossing"
    square_width: float = 20
    circle_radius: float = 4
    human_num: int = 5
    nonstop_human: bool = False
    centralized_planning: bool = True


class HumansConfig(ConfigBase):
    visible: bool = True
    policy: str = "orca"
    radius: float = 0.3
    v_pref: float = 1
    sensor: str = "coordinates"


class RobotConfig(ConfigBase):
    visible: bool = True
    policy: str = "orca_rc"
    radius: float = 0.3
    v_pref: float = 1
    sensor: str = "coordinates"


class TestConfig(BaseSettings):
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

    env: EnvironmentConfig = Field(default_factory=EnvironmentConfig)

    reward: RewardConfig = Field(default_factory=RewardConfig)

    sim: SimulationConfig = Field(default_factory=SimulationConfig)

    humans: HumansConfig = Field(default_factory=HumansConfig)

    robot: RobotConfig = Field(default_factory=RobotConfig)
