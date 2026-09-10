# Changelog

## Unreleased

- Require nonempty validation datasets with matching tuple lengths and unique
  names. Reject invalid validation setups before dataloader initialization.
- Make `TypedDictIterable` reject one-shot and unsized inputs without consuming
  them; accepted collections are preserved. Splitting still supports generators,
  and an empty dictionary now produces empty splits rather than raising `KeyError`.
- Recover the training and validation progress counters and logged values when
  continuing an existing experiment, instead of resetting them. Starting a new
  experiment from existing weights still leaves these values and the optimizer fresh.
- Keep separate randomness states for reconstructing the experimental setup and
  continuing training. Add CPU tests comparing uninterrupted and recovered runs.
- Use TorchData's `StatefulDataLoader` and save all recovery states every
  `checkpoint_interval` completed iterations (default 100), as well as at phase
  boundaries. Resume inside training or validation without repeating completed
  batches or resetting the current epoch's accumulated metrics. Require
  `torchdata>=0.11` and `torch>=2.6`.
- Atomically replace the latest checkpoint so a failed save leaves the previous
  recovery point intact, including an initial save before the first epoch.
- Save updated best-validation losses and their epochs alongside the latest
  training state. Keep the selected model's weights and epoch separately within
  the checkpoint, and restore its inference export on recovery if needed.
- Test interrupted runs, last-batch and phase boundaries, loader randomness,
  worker prefetching, and failed saves against uninterrupted training.
- Break compatibility with v0.1.x experiment recovery and remove the separate-file
  saving methods. Reject incompatible loader settings and unsupported checkpoint
  formats. Keep interrupted run directories and document the recovery assumptions.

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
