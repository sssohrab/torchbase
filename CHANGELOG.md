# Changelog

## Unreleased

- Separate sample-weighted batch metric averages (`batch_means` in TensorBoard)
  from whole-epoch metrics (`epochs`). Keep iteration values and `loss/epochs`
  unchanged; loss aggregation assumes mean-reduced batch losses. Custom layouts
  using the old metric `epochs` tags need updating.
- Write aggregate summaries once at epoch completion, including when recovering
  a checkpoint saved after the last batch but before its summary was written.
- Compute exact binary micro/macro precision, recall and F1 from bounded confusion
  counts. Keep AUC, reconstruction, segmentation and other stateless metrics as
  explicitly labeled batch statistics, without retaining predictions.
- Add the optional `EpochMetric` interface for custom accumulators, with independent
  state and reset behavior for training and each validation dataset. Preserve this
  state during mid-epoch recovery. `ValuesLogger.epoch_values` holds exact scores;
  `average_of_epoch` and `average_overall` remain averages of batch scores.
- Advance checkpoints to format 3; earlier unreleased format-2 checkpoints must
  use their original code to resume. Test uneven batches against whole-dataset
  references, dataset isolation, resets and interrupted checkpoint saves.
- Preserve unmapped metric arguments and defaults when applying partial keyword
  mappings. Handle each metric's signature separately and reject conflicting
  mappings or duplicate inputs. Test both training and validation invocation.
- Copy configuration settings without modifying the caller's dictionary or sharing
  nested values. Reject unknown top-level and session fields instead of dropping
  them when saving; retain custom settings in the supported sections.
- Name missing or invalid configuration fields, and reject values that cannot be
  saved reliably as JSON before creating a run directory. Test saved configurations
  when starting and recovering experiments; tuples still load from JSON as lists.
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
