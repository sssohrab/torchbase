# Changelog

## Unreleased

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
