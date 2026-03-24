"""Modeling session — stateful bridge between AI tool calls and geometry.

A session holds:
1. The current shape (TopoDS_Shape)
2. The operation log (what the AI did)
3. The semantic graph (what the shape means)

After every mutation, the semantic graph is rebuilt so the AI
always has an up-to-date understanding of the geometry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from OCP.TopoDS import TopoDS_Shape

from next3d.core.brep import load_step, save_step
from next3d.core.schema import SemanticGraph
from next3d.graph.semantic import build_semantic_graph_from_shape
from next3d.modeling import kernel
from next3d.modeling.operations import OpType, Operation, OperationLog


class ModelingError(Exception):
    """Raised when a modeling operation fails."""


class ModelingSession:
    """Stateful modeling session for AI-driven 3D creation and modification.

    Usage:
        session = ModelingSession()
        session.create_box(100, 60, 20)
        session.add_hole(0, 0, 10)
        graph = session.graph          # semantic understanding
        session.export_step("part.step")
        script = session.to_script()   # CadQuery Python code
    """

    def __init__(self) -> None:
        self._shape: TopoDS_Shape | None = None
        self._graph: SemanticGraph | None = None
        self._log = OperationLog()
        self._history: list[TopoDS_Shape] = []  # for undo

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> TopoDS_Shape | None:
        return self._shape

    @property
    def graph(self) -> SemanticGraph:
        """Current semantic graph. Rebuilt lazily after mutations."""
        if self._graph is None and self._shape is not None:
            self._graph = build_semantic_graph_from_shape(self._shape)
        if self._graph is None:
            return SemanticGraph()
        return self._graph

    @property
    def history(self) -> OperationLog:
        return self._log

    @property
    def is_empty(self) -> bool:
        return self._shape is None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply(self, new_shape: TopoDS_Shape, op: Operation) -> dict[str, Any]:
        """Apply a shape mutation: save undo state, update shape, log op, invalidate graph."""
        if self._shape is not None:
            self._history.append(self._shape)
        self._shape = new_shape
        self._graph = None  # invalidate — will rebuild on next .graph access
        self._log.append(op)
        # Return a summary of what happened
        g = self.graph
        return {
            "op_id": op.op_id,
            "faces": len(g.faces),
            "edges": len(g.edges),
            "features": len(g.features),
            "solids": len(g.solids),
        }

    # ------------------------------------------------------------------
    # LOAD / EXPORT
    # ------------------------------------------------------------------

    def load_step(self, path: str | Path) -> dict[str, Any]:
        """Load geometry from a STEP file."""
        model = load_step(path)
        op = Operation(
            op_type=OpType.LOAD_STEP,
            params={"path": str(path)},
            description=f"Loaded {Path(path).name}",
        )
        return self._apply(model.shape, op)

    def export_step(self, path: str | Path) -> None:
        """Export current geometry to a STEP file."""
        if self._shape is None:
            raise ModelingError("No geometry to export")
        save_step(self._shape, path)

    # ------------------------------------------------------------------
    # CREATE operations
    # ------------------------------------------------------------------

    def create_box(
        self,
        length: float,
        width: float,
        height: float,
        center: tuple[float, float, float] = (0, 0, 0),
    ) -> dict[str, Any]:
        """Create a box. Replaces current geometry."""
        shape = kernel.create_box(length, width, height, center=center)
        op = Operation(
            op_type=OpType.CREATE_BOX,
            params={"length": length, "width": width, "height": height, "center": list(center)},
            description=f"Box {length}×{width}×{height}",
        )
        return self._apply(shape, op)

    def create_cylinder(
        self,
        radius: float,
        height: float,
        center: tuple[float, float, float] = (0, 0, 0),
        axis: str = "Z",
    ) -> dict[str, Any]:
        shape = kernel.create_cylinder(radius, height, center=center, axis=axis)
        op = Operation(
            op_type=OpType.CREATE_CYLINDER,
            params={"radius": radius, "height": height, "center": list(center), "axis": axis},
            description=f"Cylinder r={radius} h={height}",
        )
        return self._apply(shape, op)

    def create_sphere(
        self,
        radius: float,
        center: tuple[float, float, float] = (0, 0, 0),
    ) -> dict[str, Any]:
        shape = kernel.create_sphere(radius, center=center)
        op = Operation(
            op_type=OpType.CREATE_SPHERE,
            params={"radius": radius, "center": list(center)},
            description=f"Sphere r={radius}",
        )
        return self._apply(shape, op)

    def create_extrusion(
        self,
        points: list[tuple[float, float]],
        height: float,
        center: tuple[float, float, float] = (0, 0, 0),
    ) -> dict[str, Any]:
        shape = kernel.create_extrusion(points, height, center=center)
        op = Operation(
            op_type=OpType.CREATE_EXTRUSION,
            params={"points": points, "height": height, "center": list(center)},
            description=f"Extruded polygon, {len(points)} vertices, h={height}",
        )
        return self._apply(shape, op)

    # ------------------------------------------------------------------
    # MODIFY operations
    # ------------------------------------------------------------------

    def _require_shape(self) -> TopoDS_Shape:
        if self._shape is None:
            raise ModelingError("No geometry loaded. Create or load a shape first.")
        return self._shape

    def add_hole(
        self,
        center_x: float,
        center_y: float,
        diameter: float,
        depth: float | None = None,
        face_selector: str = ">Z",
    ) -> dict[str, Any]:
        """Drill a hole into the current shape."""
        shape = kernel.add_hole(
            self._require_shape(), center_x, center_y, diameter, depth, face_selector
        )
        op = Operation(
            op_type=OpType.ADD_HOLE,
            params={
                "center_x": center_x, "center_y": center_y,
                "diameter": diameter, "depth": depth,
                "face_selector": face_selector,
            },
            description=f"Hole ⌀{diameter} at ({center_x},{center_y})",
        )
        return self._apply(shape, op)

    def add_counterbore_hole(
        self,
        center_x: float,
        center_y: float,
        hole_diameter: float,
        cb_diameter: float,
        cb_depth: float,
        depth: float | None = None,
        face_selector: str = ">Z",
    ) -> dict[str, Any]:
        shape = kernel.add_counterbore_hole(
            self._require_shape(), center_x, center_y,
            hole_diameter, cb_diameter, cb_depth, depth, face_selector,
        )
        op = Operation(
            op_type=OpType.ADD_COUNTERBORE_HOLE,
            params={
                "center_x": center_x, "center_y": center_y,
                "hole_diameter": hole_diameter, "cb_diameter": cb_diameter,
                "cb_depth": cb_depth, "depth": depth, "face_selector": face_selector,
            },
            description=f"Counterbore hole ⌀{hole_diameter}/⌀{cb_diameter}",
        )
        return self._apply(shape, op)

    def add_pocket(
        self,
        center_x: float,
        center_y: float,
        length: float,
        width: float,
        depth: float,
        face_selector: str = ">Z",
    ) -> dict[str, Any]:
        shape = kernel.add_pocket(
            self._require_shape(), center_x, center_y, length, width, depth, face_selector
        )
        op = Operation(
            op_type=OpType.ADD_POCKET,
            params={
                "center_x": center_x, "center_y": center_y,
                "length": length, "width": width, "depth": depth,
                "face_selector": face_selector,
            },
            description=f"Pocket {length}×{width}×{depth} at ({center_x},{center_y})",
        )
        return self._apply(shape, op)

    def add_circular_pocket(
        self,
        center_x: float,
        center_y: float,
        diameter: float,
        depth: float,
        face_selector: str = ">Z",
    ) -> dict[str, Any]:
        shape = kernel.add_circular_pocket(
            self._require_shape(), center_x, center_y, diameter, depth, face_selector
        )
        op = Operation(
            op_type=OpType.ADD_CIRCULAR_POCKET,
            params={
                "center_x": center_x, "center_y": center_y,
                "diameter": diameter, "depth": depth, "face_selector": face_selector,
            },
            description=f"Circular pocket ⌀{diameter}×{depth}",
        )
        return self._apply(shape, op)

    def add_boss(
        self,
        center_x: float,
        center_y: float,
        diameter: float,
        height: float,
        face_selector: str = ">Z",
    ) -> dict[str, Any]:
        shape = kernel.add_boss(
            self._require_shape(), center_x, center_y, diameter, height, face_selector
        )
        op = Operation(
            op_type=OpType.ADD_BOSS,
            params={
                "center_x": center_x, "center_y": center_y,
                "diameter": diameter, "height": height, "face_selector": face_selector,
            },
            description=f"Boss ⌀{diameter}×{height}",
        )
        return self._apply(shape, op)

    def add_slot(
        self,
        center_x: float,
        center_y: float,
        length: float,
        width: float,
        depth: float,
        angle: float = 0.0,
        face_selector: str = ">Z",
    ) -> dict[str, Any]:
        shape = kernel.add_slot(
            self._require_shape(), center_x, center_y,
            length, width, depth, angle, face_selector,
        )
        op = Operation(
            op_type=OpType.ADD_SLOT,
            params={
                "center_x": center_x, "center_y": center_y,
                "length": length, "width": width, "depth": depth,
                "angle": angle, "face_selector": face_selector,
            },
            description=f"Slot {length}×{width}×{depth}",
        )
        return self._apply(shape, op)

    def add_fillet(
        self,
        radius: float,
        edge_selector: str | None = None,
    ) -> dict[str, Any]:
        shape = kernel.add_fillet(self._require_shape(), radius, edge_selector)
        op = Operation(
            op_type=OpType.ADD_FILLET,
            params={"radius": radius, "edge_selector": edge_selector},
            description=f"Fillet r={radius}",
        )
        return self._apply(shape, op)

    def add_chamfer(
        self,
        distance: float,
        edge_selector: str | None = None,
    ) -> dict[str, Any]:
        shape = kernel.add_chamfer(self._require_shape(), distance, edge_selector)
        op = Operation(
            op_type=OpType.ADD_CHAMFER,
            params={"distance": distance, "edge_selector": edge_selector},
            description=f"Chamfer d={distance}",
        )
        return self._apply(shape, op)

    # ------------------------------------------------------------------
    # BOOLEAN operations
    # ------------------------------------------------------------------

    def boolean_union(self, other_session: ModelingSession) -> dict[str, Any]:
        """Union current shape with another session's shape."""
        shape = kernel.boolean_union(self._require_shape(), other_session._require_shape())
        op = Operation(
            op_type=OpType.BOOLEAN_UNION,
            params={},
            description="Boolean union",
        )
        return self._apply(shape, op)

    def boolean_cut(self, tool_shape: TopoDS_Shape) -> dict[str, Any]:
        """Cut a tool shape from the current shape."""
        shape = kernel.boolean_cut(self._require_shape(), tool_shape)
        op = Operation(
            op_type=OpType.BOOLEAN_CUT,
            params={},
            description="Boolean cut",
        )
        return self._apply(shape, op)

    # ------------------------------------------------------------------
    # TRANSFORM operations
    # ------------------------------------------------------------------

    def translate(self, dx: float = 0, dy: float = 0, dz: float = 0) -> dict[str, Any]:
        shape = kernel.translate(self._require_shape(), dx, dy, dz)
        op = Operation(
            op_type=OpType.TRANSLATE,
            params={"dx": dx, "dy": dy, "dz": dz},
            description=f"Translate ({dx},{dy},{dz})",
        )
        return self._apply(shape, op)

    def rotate(
        self,
        axis: tuple[float, float, float] = (0, 0, 1),
        angle_degrees: float = 0,
        center: tuple[float, float, float] = (0, 0, 0),
    ) -> dict[str, Any]:
        shape = kernel.rotate(self._require_shape(), axis, angle_degrees, center)
        op = Operation(
            op_type=OpType.ROTATE,
            params={"axis": list(axis), "angle_degrees": angle_degrees, "center": list(center)},
            description=f"Rotate {angle_degrees}° around {axis}",
        )
        return self._apply(shape, op)

    # ------------------------------------------------------------------
    # UNDO
    # ------------------------------------------------------------------

    def undo(self) -> dict[str, Any]:
        """Undo the last operation."""
        if not self._history:
            raise ModelingError("Nothing to undo")
        self._shape = self._history.pop()
        self._graph = None
        removed = self._log.pop()
        g = self.graph
        return {
            "undone": removed.description if removed else "",
            "faces": len(g.faces),
            "features": len(g.features),
        }

    # ------------------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------------------

    def to_script(self) -> str:
        """Export the operation log as a CadQuery Python script."""
        return self._log.to_cadquery_script()

    def summary(self) -> dict[str, Any]:
        """Get a summary of the current session state."""
        g = self.graph
        return {
            "operations": self._log.length,
            "faces": len(g.faces),
            "edges": len(g.edges),
            "vertices": len(g.vertices),
            "features": len(g.features),
            "solids": len(g.solids),
            "feature_types": [f.feature_type for f in g.features],
        }
