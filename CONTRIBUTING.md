# Development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) meeting the
minimum version in `pyproject.toml`, then run from the repository root:

```sh
uv sync --locked --group tooling
uv run --locked --group tooling ruff check torchbase tests scripts
uv run --locked coverage run -m unittest discover -s tests
uv run --locked coverage report
```

`.python-version` selects the development Python. In an IDE, use the project's
`.venv/bin/python` interpreter and ensure its uv executable meets the same
version requirement as the terminal's installation.

The tests create disposable files under `tests/storage/`; do not put experiment
data there. Coverage must meet the threshold in `pyproject.toml`. Ruff checks
syntax and undefined-name errors. See [ci.yml](.github/workflows/ci.yml) for the
current platform and Python matrix.

## Dependencies

Commit `uv.lock` so contributors and CI use reproducible dependency versions.
The `dev` group contains test dependencies; `tooling` contains Ruff and Twine.
Neither group is a runtime dependency of the published package. Downstream
projects maintain their own environment locks.

To intentionally update dependencies:

```sh
uv lock --upgrade
uv sync --locked --group tooling
```

Review the lockfile diff and run the checks before committing it. For a single
dependency, use `uv lock --upgrade-package PACKAGE`. Builds use `poetry-core`;
the Poetry CLI is not required.

On Linux, the uv development environment uses the official PyTorch CPU index.
This does not affect pip users. For a separate CUDA development environment,
install the appropriate PyTorch build following the
[PyTorch instructions](https://pytorch.org/get-started/locally/), then install
the checkout into that environment with `pip install -e .`.

## Distribution checks

Use a fresh output directory to avoid mixing old and new release files:

```sh
release_dist="$(mktemp -d)"
uv build --out-dir "$release_dist"
uv run --locked --group tooling python scripts/check_distribution.py "$release_dist"
uv run --locked --group tooling twine check --strict "$release_dist"/*
```

uv builds the wheel from the source distribution. The checker verifies metadata
and package contents using the development Python (3.11+ required). CI tests the
installed wheel outside the checkout and enforces coverage; Codecov upload is
optional.

## Changes and releases

Tie each addition or refactor to a concrete experiment need or correctness
guarantee. Prefer a short example or subclass override when sufficient. Track
planned work in GitHub issues, record changes in [CHANGELOG.md](CHANGELOG.md),
and follow [RELEASING.md](RELEASING.md) when publishing.
