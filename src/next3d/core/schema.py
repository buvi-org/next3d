"""Core data models for the semantic 3D geometry system.

These Pydantic models define the structured representation of CAD geometry
at every level: vertices, edges, faces, features, and the full semantic graph.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SurfaceType(str, Enum):
    """Analytic and freeform surface classifications."""

    PLANE = "plane"
    CYLINDER = "cylinder"
    CONE = "cone"
    SPHERE = "sphere"
    TORUS = "torus"
    BSPLINE = "bspline"
    OTHER = "other"


class CurveType(str, Enum):
    """Edge curve classifications."""

    LINE = "line"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    BSPLINE = "bspline"
    OTHER = "other"


class FeatureType(str, Enum):
    """Recognized manufacturing/design feature types."""

    THROUGH_HOLE = "through_hole"
    BLIND_HOLE = "blind_hole"
    COUNTERBORE = "counterbore"
    COUNTERSINK = "countersink"
    FILLET = "fillet"
    CHAMFER = "chamfer"
    SLOT = "slot"
    BOSS = "boss"


class RelationshipType(str, Enum):
    """Geometric relationships between entities."""

    ADJACENT = "adjacent"
    TANGENT = "tangent"
    CONCENTRIC = "concentric"
    COAXIAL = "coaxial"
    COPLANAR = "coplanar"
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    SYMMETRIC = "symmetric"
    PATTERN_MEMBER = "pattern_member"


class EdgeRelationType(str, Enum):
    """Graph edge types in the topology graph."""

    CONTAINS = "contains"
    ADJACENT = "adjacent"
    SHARES_VERTEX = "shares_vertex"


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------


class Vec3(BaseModel):
    """3D vector / point."""

    x: float
    y: float
    z: float


class UVBounds(BaseModel):
    """Parameter-space bounds for a face."""

    u_min: float
    u_max: float
    v_min: float
    v_max: float


# ---------------------------------------------------------------------------
# Topological entities
# ---------------------------------------------------------------------------


class VertexData(BaseModel):
    """A topological vertex with its 3D position."""

    persistent_id: str
    position: Vec3


class EdgeData(BaseModel):
    """A topological edge with curve geometry."""

    persistent_id: str
    curve_type: CurveType
    length: float
    start_vertex: str  # persistent_id of VertexData
    end_vertex: str  # persistent_id of VertexData
    radius: Optional[float] = None
    center: Optional[Vec3] = None


class FaceData(BaseModel):
    """A topological face with surface geometry."""

    persistent_id: str
    surface_type: SurfaceType
    area: float
    centroid: Vec3
    normal: Optional[Vec3] = None  # outward normal (for planes)
    radius: Optional[float] = None  # for cylinders, cones, spheres
    axis: Optional[Vec3] = None  # for surfaces of revolution
    uv_bounds: Optional[UVBounds] = None
    edge_ids: list[str] = Field(default_factory=list)  # persistent_ids of bounding edges


class SolidData(BaseModel):
    """A topological solid — the top-level B-Rep entity."""

    persistent_id: str
    face_ids: list[str] = Field(default_factory=list)
    volume: Optional[float] = None
    centroid: Optional[Vec3] = None


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


class FeatureData(BaseModel):
    """A recognized design/manufacturing feature."""

    persistent_id: str
    feature_type: FeatureType
    face_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    parameters: dict[str, float] = Field(default_factory=dict)
    axis: Optional[Vec3] = None
    description: str = ""


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------


class Relationship(BaseModel):
    """A geometric relationship between two entities."""

    source_id: str
    target_id: str
    relationship_type: RelationshipType
    parameters: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------


class AdjacencyEdge(BaseModel):
    """An edge in the topology graph connecting two entities."""

    source_id: str
    target_id: str
    edge_type: EdgeRelationType
    shared_edge_id: Optional[str] = None  # the edge shared between two adjacent faces


# ---------------------------------------------------------------------------
# Top-level semantic graph
# ---------------------------------------------------------------------------


class SemanticGraph(BaseModel):
    """The complete semantic representation of a 3D part — the '3D DOM'.

    This is the primary output of the system. It contains everything an AI
    agent needs to reason about a CAD part: topology, geometry, features,
    and relationships — all cross-referenced by persistent IDs.
    """

    solids: list[SolidData] = Field(default_factory=list)
    faces: list[FaceData] = Field(default_factory=list)
    edges: list[EdgeData] = Field(default_factory=list)
    vertices: list[VertexData] = Field(default_factory=list)
    features: list[FeatureData] = Field(default_factory=list)
    adjacency: list[AdjacencyEdge] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def feature_count(self) -> int:
        return len(self.features)

    def get_face(self, persistent_id: str) -> Optional[FaceData]:
        """Look up a face by persistent ID."""
        for f in self.faces:
            if f.persistent_id == persistent_id:
                return f
        return None

    def get_feature(self, persistent_id: str) -> Optional[FeatureData]:
        """Look up a feature by persistent ID."""
        for feat in self.features:
            if feat.persistent_id == persistent_id:
                return feat
        return None

    def faces_adjacent_to(self, face_id: str) -> list[str]:
        """Return persistent IDs of faces adjacent to the given face."""
        result = []
        for adj in self.adjacency:
            if adj.edge_type != EdgeRelationType.ADJACENT:
                continue
            if adj.source_id == face_id:
                result.append(adj.target_id)
            elif adj.target_id == face_id:
                result.append(adj.source_id)
        return result
