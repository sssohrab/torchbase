# Changelog

## Unreleased

- Recover the training and validation progress counters and logged values when
  continuing an existing experiment, instead of resetting them. Starting a new
  experiment from existing weights still leaves these values and the optimizer fresh.
- Keep separate randomness states for reconstructing the experimental setup and
  continuing training. Add CPU tests comparing uninterrupted and recovered runs.
- Require the saved progress to correspond to a completed training-and-validation
  epoch. Older runs without continuation randomness states can still be loaded,
  with a warning, if their other required states are present and consistent.
- Save one completed-epoch checkpoint after training, validation and model
  selection. Atomically replace it so an interrupted epoch or save leaves the
  previous recovery point intact, including an initial save before the first epoch.
- Save updated best-validation losses and their epochs alongside the latest
  training state. Keep the selected model's weights and epoch separately within
  the checkpoint, and restore its inference export on recovery if needed.
- Continue reading consistent legacy checkpoints, but use the new checkpoint in
  preference to separate state files. Stop automatic partial-epoch saves and keep
  interrupted run directories. Document replay, storage and compatibility limits.

## 0.1.4 - 2026-09-09

- Adopt standard project metadata and uv for development, with a committed
  dependency lock. Retain `poetry-core` for builds.
- Consolidate CI with conservative Ruff checks, Python 3.10–3.14 Linux tests,
  a macOS job, a current-dependency job, and tests against the built wheel.
- Validate distribution contents and release tags before publishing.
- Make the invalid-probability test deterministic to prevent intermittent failures.
- Publish tested artifacts through TestPyPI and PyPI Trusted Publishing, with
  production approval controlled by the GitHub environment.
- Document contributor commands and the recurring release process.

Training code, public imports, runtime dependency requirements, and checkpoint
behavior are unchanged.
