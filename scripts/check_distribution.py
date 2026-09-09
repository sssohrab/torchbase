"""Check release metadata and package contents using Python 3.11+ stdlib only."""

import argparse
from email.parser import BytesParser
from pathlib import Path
import re
import tarfile
import tomllib
import zipfile


def check_distribution(dist_dir: Path, tag: str | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if tag is not None:
        if not re.fullmatch(r"v\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", tag):
            raise ValueError("Use a version tag such as v0.1.4 or v0.1.4rc1.")
        if tag != f"v{version}":
            raise ValueError(f"Tag {tag!r} does not match package version {version!r}.")

    wheels = list(dist_dir.glob("*.whl"))
    sdists = list(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one sdist; use a fresh output directory.")

    sources = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / "torchbase").rglob("*.py")
    }
    with zipfile.ZipFile(wheels[0]) as wheel:
        metadata_path, = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        metadata = BytesParser().parsebytes(wheel.read(metadata_path))
        packaged_sources = {name: wheel.read(name) for name in wheel.namelist() if name.endswith(".py")}
        if packaged_sources != sources:
            raise ValueError("Wheel Python sources do not match torchbase/ exactly.")
        if not any(name.endswith("/LICENSE") for name in wheel.namelist()):
            raise ValueError("Wheel is missing LICENSE.")

    with tarfile.open(sdists[0]) as sdist:
        prefix = f"{project['name']}-{version}/"
        for name, content in sources.items():
            with sdist.extractfile(prefix + name) as source:
                if source.read() != content:
                    raise ValueError(f"Sdist source differs: {name}")
        for name in ("README.md", "LICENSE", "pyproject.toml"):
            with sdist.extractfile(prefix + name) as source:
                if source.read() != (root / name).read_bytes():
                    raise ValueError(f"Sdist file differs: {name}")
        with sdist.extractfile(prefix + "PKG-INFO") as source:
            sdist_metadata = BytesParser().parsebytes(source.read())

    for artifact_metadata in (metadata, sdist_metadata):
        for field, expected in (
            ("Name", project["name"]),
            ("Version", version),
            ("Requires-Python", project["requires-python"]),
        ):
            if artifact_metadata[field] != expected:
                raise ValueError(f"Unexpected {field}: {artifact_metadata[field]!r}")
        if set(artifact_metadata.get_all("Requires-Dist", [])) != set(project["dependencies"]):
            raise ValueError("Published dependencies differ from [project].dependencies.")
        if artifact_metadata.get_payload().strip() != (root / "README.md").read_text().strip():
            raise ValueError("Package description differs from README.md.")

    print(f"Verified {wheels[0].name} and {sdists[0].name}: metadata, sources, README, license.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path)
    parser.add_argument("--tag", help="Require a conventional tag matching the package version")
    args = parser.parse_args()
    check_distribution(args.dist_dir, args.tag)
