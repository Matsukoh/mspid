# MSPID: Multi-Source Policy Improvement Distillation for Unified Imitation and Reinforcement Learning

## Evaluating trained SocialNav models

All five SocialNav scripts support evaluation without training. Pass `--eval.run-path` to load a saved model and evaluate it on the `test` scenario. Without this option, the scripts train as usual.

```bash
python run_mspid_socialnav.py --eval.run-path ./wandb/run-XXXX
python run_mspid_rmflow_socialnav.py --eval.run-path ./wandb/run-YYYY
python run_nclql_socialnav.py --eval.run-path ./wandb/run-ZZZZ
python run_qsm_socialnav.py --eval.run-path ./wandb/run-QSM
python run_sac_socialnav.py --eval.run-path ./wandb/run-SAC
```

The path can point to a local W&B run directory, its `files/` or `trained_models/` directory, or a `.pth` checkpoint file. For directory paths, the default checkpoint is `model_best.pth` in the corresponding `trained_models/` directory. Downloading runs from online W&B URLs or run IDs is not supported.

```bash
python run_mspid_socialnav.py \
  --eval.run-path ./wandb/run-XXXX \
  --eval.checkpoint model_10000.pth \
  --eval.episodes 100 \
  --eval.device cpu \
  --eval.output-dir ./evaluations/mspid_10000
```

- `--eval.checkpoint`: Checkpoint filename when a directory is provided. Defaults to `model_best.pth`.
- `--eval.episodes`: Number of test episodes. Defaults to `env.test_size` from the saved configuration.
- `--eval.device`: `auto` (default), `cpu`, `cuda`, or `mps`.
- `--eval.render --eval.render-type video`: Save evaluation videos. Use `traj` to save trajectory images instead.
- `--eval.output-dir`: Output directory for evaluation results in CSV, TXT, and JSON format, the evaluation configuration, and rendered outputs. Defaults to `evaluations/<checkpoint_name>_<timestamp>/`.
- `--eval.config-path`: Explicit path to the training configuration in JSON format or a W&B `config.yaml` file.

The saved training configuration is the baseline for the model, environment, observation transformation, and random seed. Explicit CLI arguments override individual fields, while unspecified fields retain their saved values. CLI defaults, environment variables, and `.env` values do not override these saved fields. Explicit values equal to defaults and negative boolean flags (for example, `--no-env.randomize-attributes`) are supported. Nested JSON options are merged field by field as well.

```bash
python run_mspid_socialnav.py \
  --eval.run-path ./wandb/run-XXXX \
  --sim.human-num 10 \
  --sim.circle-radius 6 \
  --env.time-limit 40 \
  --train.random-seed 23 \
  --eval.output-dir ./evaluations/humans_10
```

The same options work for all five scripts. `eval.*` controls use the current invocation's settings, and evaluation always disables W&B logging and model saving. The effective configuration is saved in `config.json`, and explicit CLI arguments are recorded in `results.json` under `cli_overrides`. Changes to model dimensions must remain compatible with the saved weights; incompatible weights fail strict loading. Changing the reward configuration also changes the meaning of reward-based metrics such as CDR.

Configuration sources are checked in this order: an explicitly supplied configuration file, the checkpoint's embedded `config`, then `config.json` or `config.yaml` in the checkpoint directory or its parent. For legacy checkpoints without an available configuration, supply `--eval.config-path`. Copied `config.py` files are not loaded because they do not capture CLI overrides used during training. Legacy actor checkpoints saved after compilation are also supported.

New checkpoints saved with `--log.save-model` include the training configuration, algorithm name, and training step alongside the model weights. When W&B is disabled, checkpoints are saved to `models/<timestamp>_<algorithm>/trained_models/`.

Run the evaluation tests with:

```bash
python -m unittest discover -s tests -p 'test_socialnav_checkpoints.py' -v
```

## Uploading the best model to W&B

With `--log.wandb`, each training script saves the model with the highest validation CDR after training and registers only that checkpoint for upload as `trained_models/model_best.pth` in the run's Files tab. Upload is scheduled for run completion, after the final test evaluation. The checkpoint includes the training configuration, algorithm, best training step, and validation CDR. In offline mode, it is uploaded when the run is synced later.

```bash
python run_mspid_socialnav.py --log.wandb
```

Add `--log.save-model` to retain intermediate checkpoints locally. With W&B enabled, these are stored in `<run_directory>/local_checkpoints/trained_models/`, outside the synced `files/` directory. They are not uploaded. Without W&B, local checkpoints continue to use `models/<timestamp>_<algorithm>/trained_models/`. Evaluation can resolve both the uploaded best checkpoint and local intermediate checkpoints from `--eval.run-path <run_directory>`.

## Training QSM on SocialNav

`run_qsm_socialnav.py` uses the same robot-frame observations, GAT aggregation,
ORCA preliminary exploration, episode-based updates, validation CDR selection,
and checkpoint format as the MSPID SocialNav runner. The QSM algorithm is in
`socialnav/trainer.py` (`SocialQSMTrainer`); its diffusion policy and networks are
in `diffusion/`. Settings are in `configs/qsm_socialnav_config.py`, with the same
nested CLI and `EXPERIMENT_` environment-variable conventions as MSPID.

```bash
python run_qsm_socialnav.py --log.save-model --train.random-seed 17
python run_qsm_socialnav.py --log.wandb --log.save-model
python run_qsm_socialnav.py --eval.run-path ./models/<run>/trained_models/model_best.pth \
  --eval.episodes 100 --eval.device cpu
```

The legacy QSM objective is retained: the diffusion network predicts
`-M * (mean(dQ1/da, dQ2/da) + eta)` at forward-noised replay actions. The TD target
uses the minimum of the two target Q values and a target diffusion actor; both
actor and critic targets receive Polyak updates. Unlike NC-LQL, QSM's critic is
not conditioned on diffusion time. `--train.m` defaults to 50, `--train.gamma`
to 0.99, and `--train.polyak` to 0.995. `--no-train.use-eta` disables the legacy
1e-16 offset. Each training episode performs one update, matching the current
MSPID runner.

The diffusion defaults retain 100 steps, a linear beta schedule, Mish networks,
and the legacy reverse-sampling noise multiplier of 0.2. Adjust these with
`--model.n-timesteps`, `--model.beta-schedule` (`linear`, `cosine`, `vp`), and
`--model.sampling-noise-scale`. `--no-model.random-sample` starts from zero and
disables reverse-step noise. Action bounds are per-dimension buffers that move
with the policy to CPU/CUDA/MPS. Forward noising and sampling keep the batch
axis even for batch size 1. Sampling runs without retaining autograd graphs.

Validation also runs at the final episode so short runs produce a best model
when `total_it` is below `eval_interval`. The shared saved-run evaluator supports
QSM checkpoints produced by this runner; old `legacy` checkpoints require
conversion because their observation conventions and model layouts differ.
For comparisons, align environment/reward settings, seeds, preliminary episodes,
training episodes, and validation/test sizes across algorithms. Full-length
learning performance has not been verified by the smoke tests.

```bash
python -m unittest discover -s tests -p 'test_qsm.py' -v
```

## Training SAC on SocialNav

`run_sac_socialnav.py` trains Soft Actor-Critic with automatic entropy tuning.
It uses the common robot-frame observations, separate actor/critic GAT encoders,
ORCA preliminary exploration, validation CDR selection, and saved-run evaluation.
The Gaussian policy and twin Q heads are in `sac/models.py`, the algorithm is
`SocialSACTrainer` in `socialnav/trainer.py`, and settings are in
`configs/sac_socialnav_config.py`.

```bash
python run_sac_socialnav.py --log.save-model --train.random-seed 17
python run_sac_socialnav.py --log.wandb --log.save-model
python run_sac_socialnav.py --eval.run-path ./models/<run>/trained_models/model_best.pth \
  --eval.episodes 100 --eval.device cpu
```

The policy uses reparameterized Gaussian samples, tanh squashing, and an affine
transform to per-dimension action bounds. Log probabilities include both
transform Jacobians and use a stable correction for saturated tanh outputs.
The critic target is `reward + gamma * (1 - done) * (min(target_Q1, target_Q2)
- alpha * next_log_prob)`. The actor minimizes `alpha * log_prob - min(Q1, Q2)`.
Only the critic has a Polyak target network; there is no target actor.

`log_alpha` is optimized automatically using the detached policy log probability.
`--train.init-alpha` defaults to 0.2, `--train.alpha-lr` to 0.0003, and
`--train.target-entropy` defaults to minus the action dimension (-2 here).
Entropy is measured with respect to the configured action coordinates, including
scaling. W&B records actor/critic/temperature losses, `train/alpha`, and
`train/entropy`. Checkpoints include `trainer_state_dict` with the learned
`log_alpha`, target entropy, alpha optimizer state, and target critic weights.
Evaluation only needs the actor weights; this runner does not provide a training
resume CLI.

Validation and test evaluation use the squashed Gaussian mean by default.
Use `--no-eval.deterministic` for stochastic evaluation. Training always samples
stochastically. `--train.lr` defaults to 0.0003, `--train.gamma` to 0.99, and
`--train.polyak` to 0.995 (the retained target weight).
`--train.updates-per-episode` defaults to 1 to match the existing SocialNav
runners; increase it explicitly for more replay updates. Preliminary ORCA
exploration stores transitions without gradient updates. Align episode counts,
update budgets, environment settings, and seeds when comparing methods.
Terminal transitions, including timeouts, use the existing SocialNav `done`
convention and do not bootstrap. Validation also runs at the final episode.

CPU smoke tests cover short training, model saving/loading, evaluation,
action-density corrections, terminal masking, temperature update direction,
and Polyak updates. Full-length learning performance has not been verified.

```bash
python -m unittest discover -s tests -p 'test_sac.py' -v
```
