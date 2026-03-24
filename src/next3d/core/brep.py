"""B-Rep ingestion: STEP file → OpenCascade topology.

This module is the entry point for loading CAD files. It wraps OpenCascade's
STEP reader and exposes the resulting shape for downstream processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopoDS import TopoDS_Shape


class STEPLoadError(Exception):
    """Raised when a STEP file cannot be parsed."""


@dataclass(frozen=True)
class BRepModel:
    """A loaded B-Rep model wrapping an OpenCascade shape."""

    shape: TopoDS_Shape
    source_path: Path

    @property
    def is_null(self) -> bool:
        return self.shape.IsNull()


def load_step(path: str | Path) -> BRepModel:
    """Load a STEP file and return the B-Rep model.

    Args:
        path: Path to a STEP (AP203/AP214) file.

    Returns:
        BRepModel containing the parsed OpenCascade shape.

    Raises:
        STEPLoadError: If the file cannot be read or contains no geometry.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"STEP file not found: {path}")

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))

    if status != IFSelect_RetDone:
        raise STEPLoadError(f"Failed to read STEP file: {path} (status={status})")

    reader.TransferRoots()
    shape = reader.OneShape()

    if shape.IsNull():
        raise STEPLoadError(f"STEP file contains no geometry: {path}")

    return BRepModel(shape=shape, source_path=path)
