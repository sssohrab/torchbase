"""Single-file checkpoint writes that leave the previous save intact on failure."""

import os
import tempfile

import torch


def atomic_torch_save(state, path: str) -> None:
    # The temporary file must be on the same filesystem as its destination.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=os.path.dirname(path),
                                         prefix=".checkpoint-", suffix=".tmp", delete=False) as file:
            temporary_path = file.name
            torch.save(state, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
