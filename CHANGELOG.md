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
- Explain the current recovery assumptions and limitations in the README.
  Preserving a complete checkpoint after an arbitrary interruption and correcting
  the saved best-model metadata remain part of #32.

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
