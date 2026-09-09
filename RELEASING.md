# Releases

The [publish workflow](.github/workflows/publish.yml) runs CI on a version tag,
uploads the tested distributions to TestPyPI, then publishes the same files to
PyPI after environment approval.

## Required configuration

These settings live outside Git and must remain consistent with the workflow:

- GitHub environments `testpypi` and `pypi` allow deployment from tags matching
  `v*`. The `pypi` environment requires a maintainer's approval. Without that
  setting, successful tag builds can publish to production automatically.
- Each index has a Trusted Publisher for `sssohrab/torchbase`, workflow filename
  `publish.yml`, and its matching environment name (`testpypi` or `pypi`).

For setup or recovery, see the official
[GitHub environment documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments)
and [PyPI Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/).

## Versioning

Use patch releases for compatible fixes, tooling, and documentation changes.
While the project is pre-1.0, use a minor release for changes to public interfaces,
checkpoint formats, or experiment semantics that require migration guidance.
Prereleases such as `0.2.0rc1` allow opt-in downstream testing.

Keep published versions and tags immutable. Downstream projects should pin
their chosen version and maintain their own dependency locks.

## Preparing a release

1. Update `[project].version` in `pyproject.toml` and run `uv lock`. Move the
   relevant `CHANGELOG.md` entries into a dated, versioned section.
2. Run the checks in [CONTRIBUTING.md](CONTRIBUTING.md), review the distributions,
   and merge the release change after CI passes. Exercise a representative
   downstream project before a stable release.
3. Create a matching tag on that reviewed commit and push it explicitly.
   For example, **only when the package version is `0.1.4`**:

   ```sh
   git tag -a v0.1.4 -m "Release torchbase 0.1.4"
   git push origin v0.1.4
   ```

   Tags use `v` followed by the exact package version, without `-test` or `-final`.
4. After CI and the TestPyPI upload succeed, inspect the `distributions` artifact
   and test the staged package in a disposable environment. For the example
   version above, install dependencies from PyPI, then only torchbase from TestPyPI:

   ```sh
   python -m pip install torch datasets tensorboard numpy scikit-learn
   python -m pip install --no-deps --index-url https://test.pypi.org/simple/ torchbase==0.1.4
   python -c "from torchbase import TrainingBaseSession; print(TrainingBaseSession)"
   ```

   Run the import check outside the repository. Avoid combining the two indexes
   with `--extra-index-url` for dependency resolution.
5. Approve the waiting `pypi` deployment in GitHub Actions.
6. Create GitHub release notes from the changelog. Mark prereleases as such in
   GitHub too; the package version, rather than that checkbox, controls pip's
   prerelease handling.

## Failed releases

If production publishing fails after TestPyPI succeeds, rerun only the failed
jobs. A complete rerun attempts the existing TestPyPI upload again and fails
rather than silently skipping it. If an index accepted some files before an
upload failed, inspect the release before retrying: existing files cannot be
replaced, and the workflow does not skip duplicate uploads.

Fix a broken release with a new version. If necessary, yank the broken release
on PyPI with an explanation instead of deleting it.
