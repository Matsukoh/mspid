# MSPID: Multi-Source Policy Improvement Distillation for Unified Imitation and Reinforcement Learning

## Evaluating trained SocialNav models

All three scripts support evaluation without training. Pass `--eval.run-path` to load a saved model and evaluate it on the `test` scenario. Without this option, the scripts train as usual.

```bash
python run_mspid_socialnav.py --eval.run-path ./wandb/run-XXXX
python run_mspid_rmflow_socialnav.py --eval.run-path ./wandb/run-YYYY
python run_nclql_socialnav.py --eval.run-path ./wandb/run-ZZZZ
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

The model, environment, observation transformation, and random seed use the saved training configuration. Use `eval.*` CLI options to control evaluation; training options such as `model.*` and `env.*` do not override the saved configuration. Evaluation saves results locally without creating a W&B run.

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
