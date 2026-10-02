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

    obs_dim: int = 4
    r_obs_dim: int = 5
    act_dim: int = 2
    action_space_high: list[float] = [1.0, 1.0]
    action_space_low: list[float] = [-1.0, -1.0]


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


class TransfuncConfig(ConfigBase):
    with_peds_vel: bool = True
    peds_vel_as_relative: bool = True
    use_omega: bool = True


class EvaluationConfig(ConfigBase):
    deterministic: bool = True
    run_path: str | None = None
    checkpoint: str = "model_best.pth"
    config_path: str | None = None
    output_dir: str | None = None
    episodes: int | None = Field(default=None, gt=0)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    eval_interval: int = Field(default=1000, gt=0)
    final_eval_num: int = 500
    val_render: bool = False
    render: bool = False
    render_type: str = "video"


class ModelConfig(ConfigBase):
    h_dims: list[int] = Field(default=[256, 256], min_length=1)
    projection_dim: int = Field(default=32, gt=0, multiple_of=8)
    aggregator_enc_hdims: list[int] = [64]
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    @model_validator(mode="after")
    def validate_std(self):
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be below log_std_max")
        return self


class TrainConfig(ConfigBase):
    training_alg: Literal["SAC"] = "SAC"
    random_seed: int = 17
    lr: float = Field(default=3e-4, gt=0)
    alpha_lr: float = Field(default=3e-4, gt=0)
    init_alpha: float = Field(default=0.2, gt=0)
    target_entropy: float | None = None
    buffer_capacity: int = Field(default=1000000, gt=0)
    batch_size: int = Field(default=256, gt=0)
    preliminary_exp_n: int = Field(default=2000, ge=0)
    total_it: int = Field(default=100000, gt=0)
    polyak: float = Field(default=0.995, ge=0, le=1)
    gamma: float = Field(default=0.99, ge=0, le=1)
    # updates_per_episode: int = Field(default=1, gt=0)


class LogConfig(ConfigBase):
    wandb_project: str = "SAC"
    # wandb_mode: str = "offline"
    wandb_mode: str = "online"
    wandb: bool = False
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

    env: EnvironmentConfig = Field(default_factory=EnvironmentConfig)

    reward: RewardConfig = Field(default_factory=RewardConfig)

    sim: SimulationConfig = Field(default_factory=SimulationConfig)

    humans: HumansConfig = Field(default_factory=HumansConfig)

    robot: RobotConfig = Field(default_factory=RobotConfig)

    transfunc: TransfuncConfig = Field(default_factory=TransfuncConfig)

    eval: EvaluationConfig = Field(default_factory=EvaluationConfig)

    model: ModelConfig = Field(default_factory=ModelConfig)

    train: TrainConfig = Field(default_factory=TrainConfig)

    log: LogConfig = Field(default_factory=LogConfig)
