"""The Flag envelope: the one loose contract every check module speaks.

Design intent:

- The *envelope* (check_id, scan_id, severity, explanation, location) is stable.
- The *payload* is an OPEN tagged union discriminated by ``kind``. A new check
  adds a payload variant; it never has to change the envelope.
- ``explanation`` is always plaintext so a flag is human-readable even with no
  renderable payload.
- Payloads that carry voxel data (masks, heatmaps, meshes) do NOT inline the
  array. They reference a ``resource`` key that the server streams as a file.
  This keeps the JSON small and lets the viewer fetch overlays lazily.

The viewer renders purely by ``payload.kind`` and never asks who produced the
flag - so an agent- or NeuroJEPA-authored flag drops into the same surface with
no rework.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

# Rank used to sort flags for the adjudication queue (higher = more urgent).
_SEVERITY_RANK = {"info": 0, "warn": 1, "error": 2, "critical": 3}


class Severity(str, Enum):
    info = "info"
    warn = "warn"
    error = "error"
    critical = "critical"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self.value]


# --- Payload variants (open tagged union, discriminated by ``kind``) ----------


class NonePayload(BaseModel):
    """No renderable geometry; the flag lives entirely in its plaintext."""

    kind: Literal["none"] = "none"


class MaskPayload(BaseModel):
    """A binary/label volume drawn as a coloured overlay on the base scan."""

    kind: Literal["mask"] = "mask"
    resource: str  # key -> a NIfTI mask served by /api/resource/{key}
    colormap: str = "red"
    opacity: float = 0.5
    label: Optional[str] = None


class MeshPayload(BaseModel):
    """A surface (typically marching-cubes of a mask) drawn in the 3D render."""

    kind: Literal["mesh"] = "mesh"
    resource: str  # key -> a mesh file (OBJ, world/mm coords) served by the API
    rgba: list[float] = Field(default_factory=lambda: [1.0, 0.3, 0.3, 1.0])
    opacity: float = 1.0


class HeatmapPayload(BaseModel):
    """A continuous scalar field (e.g. anomaly score) drawn as a hot overlay."""

    kind: Literal["heatmap"] = "heatmap"
    resource: str
    colormap: str = "warm"
    opacity: float = 0.6
    cal_min: Optional[float] = None
    cal_max: Optional[float] = None


class PointPayload(BaseModel):
    """A labelled marker at a world/mm coordinate."""

    kind: Literal["point"] = "point"
    coord_mm: list[float]
    text: str = ""
    rgba: list[float] = Field(default_factory=lambda: [1.0, 1.0, 0.0, 1.0])


class Marker(BaseModel):
    coord_mm: list[float]
    text: str = ""
    rgba: list[float] = Field(default_factory=lambda: [1.0, 1.0, 0.0, 1.0])


class PointsPayload(BaseModel):
    """Several labelled markers at once (e.g. an L/R laterality pair)."""

    kind: Literal["points"] = "points"
    markers: list[Marker]


class BBoxPayload(BaseModel):
    """An axis-aligned box in world/mm space (min/max corners)."""

    kind: Literal["bbox"] = "bbox"
    min_mm: list[float]
    max_mm: list[float]
    text: str = ""
    rgba: list[float] = Field(default_factory=lambda: [0.2, 0.8, 1.0, 1.0])


Payload = Annotated[
    Union[
        NonePayload,
        MaskPayload,
        MeshPayload,
        HeatmapPayload,
        PointPayload,
        PointsPayload,
        BBoxPayload,
    ],
    Field(discriminator="kind"),
]


# --- Envelope -----------------------------------------------------------------


class Location(BaseModel):
    """Where to send the camera. ``world_mm`` drives the viewer fly-to."""

    world_mm: Optional[list[float]] = None
    voxel: Optional[list[int]] = None


class Flag(BaseModel):
    check_id: str
    scan_id: str
    severity: Severity
    explanation: str  # always plaintext, always present
    location: Optional[Location] = None
    payload: Payload = Field(default_factory=NonePayload)
    # Open bag for module-specific structured data the viewer may show in a
    # panel (e.g. a histogram, an affine matrix). Never part of the contract.
    extra: dict = Field(default_factory=dict)

    def sort_key(self) -> tuple[int, str]:
        return (-self.severity.rank, self.check_id)
