"""Modeling session — stateful bridge between AI tool calls and geometry.

A session holds:
1. Named bodies (multi-body support) — each body is a TopoDS_Shape
2. The operation log (what the AI did)
3. Semantic graphs per body
4. Assembly placements and mate constraints

After every mutation, the semantic graph is rebuilt so the AI
always has an up-to-date understanding of the geometry.

Backward compatible: single-body workflows use an implicit "default" body.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from OCP.TopoDS import TopoDS_Shape, TopoDS_Compound, TopoDS_Builder

from next3d.core.brep import load_step, save_step
from next3d.core.schema import SemanticGraph
from next3d.graph.semantic import build_semantic_graph_from_shape
from next3d.modeling import kernel
from next3d.modeling.operations import OpType, Operation, OperationLog
from next3d.modeling.parametric import ParametricEngine


class ModelingError(Exception):
    """Raised when a modeling operation fails."""


DEFAULT_BODY = "default"


@dataclass
class Placement:
    """Position and orientation of a body in assembly space."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    axis_x: float = 0.0
    axis_y: float = 0.0
    axis_z: float = 1.0
    angle_degrees: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "translation": {"x": self.x, "y": self.y, "z": self.z},
            "rotation": {
                "axis": [self.axis_x, self.axis_y, self.axis_z],
                "angle_degrees": self.angle_degrees,
            },
        }


@dataclass
class MateConstraint:
    """A declarative constraint between two bodies."""

    mate_type: str  # coincident, concentric, flush, distance, angle
    body_a: str
    entity_a: str  # persistent_id
    body_b: str
    entity_b: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mate_type": self.mate_type,
            "body_a": self.body_a,
            "entity_a": self.entity_a,
            "body_b": self.body_b,
            "entity_b": self.entity_b,
            "parameters": self.parameters,
        }


class ModelingSession:
    """Stateful modeling session for AI-driven 3D creation and modification.

    Supports multiple named bodies, assembly placement, and interference checking.

    Single-body usage (backward compatible):
        session = ModelingSession()
        session.create_box(100, 60, 20)       # creates "default" body
        session.add_hole(0, 0, 10)             # modifies active body
        session.export_step("part.step")

    Multi-body usage:
        session = ModelingSession()
        session.create_body("bracket", "box", length=100, width=60, height=20)
        session.create_body("shaft", "cylinder", radius=5, height=80)
        session.set_active_body("bracket")
        session.add_hole(0, 0, 12)             # drills into bracket
        session.place_body("shaft", x=0, y=0, z=20)
    """

    def __init__(self) -> None:
        # Multi-body registry
        self._bodies: dict[str, TopoDS_Shape] = {}
        self._active_body: str = DEFAULT_BODY
        self._body_materials: dict[str, str] = {}  # body name → material key
        self._body_metadata: dict[str, dict[str, Any]] = {}

        # Per-body state
        self._graphs: dict[str, SemanticGraph | None] = {}
        self._histories: dict[str, list[TopoDS_Shape]] = {}

        # Assembly state
        self._placements: dict[str, Placement] = {}
        self._mates: list[MateConstraint] = []

        # Parametric engine (full: bindings, dependency graph, selective replay)
        self._parametric = ParametricEngine()
        self._parameters: dict[str, dict[str, Any]] = {}  # legacy compat

        # GD&T annotations per body
        self._gdt: dict[str, Any] = {}  # body name → GDTAnnotationSet

        # Active sketch (None when no sketch in progress)
        self._active_sketch = None

        # Topology optimization state
        self._loads: list[Any] = []
        self._boundary_conditions: list[Any] = []

        # Global operation log
        self._log = OperationLog()

    # ------------------------------------------------------------------
    # Properties (backward compatible)
    # ------------------------------------------------------------------

    @property
    def shape(self) -> TopoDS_Shape | None:
        """Current active body's shape. Backward compatible."""
        return self._bodies.get(self._active_body)

    @property
    def _shape(self) -> TopoDS_Shape | None:
        """Alias for backward compatibility with internal code."""
        return self.shape

    @property
    def graph(self) -> SemanticGraph:
        """Semantic graph of the active body. Rebuilt lazily."""
        name = self._active_body
        shape = self._bodies.get(name)
        if shape is None:
            return SemanticGraph()
        if self._graphs.get(name) is None:
            self._graphs[name] = build_semantic_graph_from_shape(shape)
        return self._graphs[name]

    @property
    def history(self) -> OperationLog:
        return self._log

    @property
    def is_empty(self) -> bool:
        return len(self._bodies) == 0

    @property
    def active_body_name(self) -> str:
        return self._active_body

    @property
    def body_names(self) -> list[str]:
        return list(self._bodies.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply(self, new_shape: TopoDS_Shape, op: Operation) -> dict[str, Any]:
        """Apply a shape mutation to the active body."""
        name = self._active_body

        # Save undo state
        old_shape = self._bodies.get(name)
        if old_shape is not None:
            self._histories.setdefault(name, []).append(old_shape)

        self._bodies[name] = new_shape
        self._graphs[name] = None  # invalidate
        self._log.append(op)

        # Record in parametric engine (captures @param bindings)
        op_index = self._log.length - 1
        self._parametric.record_operation(
            op_index, op.op_type, op.params, op.description,
        )

        g = self.graph
        result = {
            "op_id": op.op_id,
            "body": name,
            "faces": len(g.faces),
            "edges": len(g.edges),
            "features": len(g.features),
            "solids": len(g.solids),
        }
        if len(self._bodies) > 1:
            result["total_bodies"] = len(self._bodies)
        return result

    def _require_shape(self) -> TopoDS_Shape:
        shape = self._bodies.get(self._active_body)
        if shape is None:
            raise ModelingError("No geometry loaded. Create or load a shape first.")
        return shape

    # ------------------------------------------------------------------
    # MULTI-BODY operations
    # ------------------------------------------------------------------

    def create_body(
        self,
        name: str,
        shape_type: str,
        material: str = "steel",
        **params: Any,
    ) -> dict[str, Any]:
        """Create a named body and make it active.

        Args:
            name: Body name (must be unique).
            shape_type: "box", "cylinder", "sphere", "extrusion".
            material: Material key (steel, aluminum, abs, etc.).
            **params: Shape parameters (length, width, height, radius, etc.).
        """
        if name in self._bodies:
            raise ModelingError(f"Body '{name}' already exists. Delete it first or use a different name.")

        creators = {
            "box": kernel.create_box,
            "cylinder": kernel.create_cylinder,
            "sphere": kernel.create_sphere,
            "extrusion": kernel.create_extrusion,
        }
        creator = creators.get(shape_type)
        if creator is None:
            raise ModelingError(f"Unknown shape type: {shape_type}. Use: {', '.join(creators)}")

        shape = creator(**params)
        old_active = self._active_body
        self._active_body = name
        self._body_materials[name] = material

        op = Operation(
            op_type=OpType.CREATE_NAMED_BODY,
            params={"name": name, "shape_type": shape_type, "material": material, **params},
            description=f"Body '{name}' ({shape_type})",
        )
        return self._apply(shape, op)

    def set_active_body(self, name: str) -> dict[str, Any]:
        """Switch which body subsequent operations target."""
        if name not in self._bodies:
            raise ModelingError(f"Body '{name}' not found. Available: {', '.join(self._bodies)}")
        self._active_body = name
        g = self.graph
        return {
            "active_body": name,
            "faces": len(g.faces),
            "features": len(g.features),
            "solids": len(g.solids),
        }

    def list_bodies(self) -> dict[str, Any]:
        """List all bodies with summary info."""
        from next3d.core.properties import MATERIALS, compute_physical_properties

        bodies = []
        for name, shape in self._bodies.items():
            g = self._graphs.get(name)
            if g is None:
                g = build_semantic_graph_from_shape(shape)
                self._graphs[name] = g

            mat_key = self._body_materials.get(name, "steel")
            density = MATERIALS.get(mat_key, 0.00785)

            try:
                props = compute_physical_properties(shape, density)
                volume = props.volume
                mass = props.mass
            except Exception:
                volume = 0
                mass = 0

            bodies.append({
                "name": name,
                "active": name == self._active_body,
                "material": mat_key,
                "faces": len(g.faces),
                "features": len(g.features),
                "volume_mm3": round(volume, 2),
                "mass_grams": round(mass, 2),
                "placement": self._placements.get(name, Placement()).to_dict(),
            })

        return {
            "count": len(bodies),
            "bodies": bodies,
            "active_body": self._active_body,
        }

    def delete_body(self, name: str) -> dict[str, Any]:
        """Delete a named body."""
        if name not in self._bodies:
            raise ModelingError(f"Body '{name}' not found.")
        del self._bodies[name]
        self._graphs.pop(name, None)
        self._histories.pop(name, None)
        self._body_materials.pop(name, None)
        self._placements.pop(name, None)
        # Remove mates referencing this body
        self._mates = [m for m in self._mates if m.body_a != name and m.body_b != name]

        # Switch active to another body if current was deleted
        if self._active_body == name:
            self._active_body = next(iter(self._bodies), DEFAULT_BODY)

        op = Operation(
            op_type=OpType.DELETE_BODY,
            params={"name": name},
            description=f"Deleted body '{name}'",
        )
        self._log.append(op)
        return {"deleted": name, "remaining_bodies": list(self._bodies.keys())}

    def duplicate_body(self, source: str, new_name: str) -> dict[str, Any]:
        """Duplicate an existing body under a new name."""
        if source not in self._bodies:
            raise ModelingError(f"Body '{source}' not found.")
        if new_name in self._bodies:
            raise ModelingError(f"Body '{new_name}' already exists.")

        import cadquery as cq
        # OCC shapes can't be deepcopied; use CadQuery's copy
        self._bodies[new_name] = cq.Shape(self._bodies[source]).copy().wrapped
        self._body_materials[new_name] = self._body_materials.get(source, "steel")
        self._graphs[new_name] = None

        old_active = self._active_body
        self._active_body = new_name
        g = self.graph
        self._active_body = old_active

        return {
            "source": source,
            "new_name": new_name,
            "faces": len(g.faces),
            "total_bodies": len(self._bodies),
        }

    def boolean_bodies(
        self,
        operation: str,
        body_a: str,
        body_b: str,
        result_name: str | None = None,
    ) -> dict[str, Any]:
        """Boolean operation between two named bodies."""
        if body_a not in self._bodies:
            raise ModelingError(f"Body '{body_a}' not found.")
        if body_b not in self._bodies:
            raise ModelingError(f"Body '{body_b}' not found.")

        ops = {"union": kernel.boolean_union, "cut": kernel.boolean_cut, "intersect": kernel.boolean_intersect}
        op_fn = ops.get(operation)
        if op_fn is None:
            raise ModelingError(f"Unknown operation: {operation}. Use: union, cut, intersect")

        result_shape = op_fn(self._bodies[body_a], self._bodies[body_b])
        target = result_name or body_a
        self._bodies[target] = result_shape
        self._graphs[target] = None

        op = Operation(
            op_type=OpType.BOOLEAN_BODIES,
            params={"operation": operation, "body_a": body_a, "body_b": body_b, "result_name": target},
            description=f"Boolean {operation}: {body_a} + {body_b} → {target}",
        )
        self._log.append(op)

        self._active_body = target
        g = self.graph
        return {
            "result_body": target,
            "faces": len(g.faces),
            "features": len(g.features),
            "total_bodies": len(self._bodies),
        }

    # ------------------------------------------------------------------
    # ASSEMBLY operations
    # ------------------------------------------------------------------

    def place_body(
        self,
        name: str,
        x: float = 0,
        y: float = 0,
        z: float = 0,
        axis_x: float = 0,
        axis_y: float = 0,
        axis_z: float = 1,
        angle_degrees: float = 0,
    ) -> dict[str, Any]:
        """Position a body in assembly space."""
        if name not in self._bodies:
            raise ModelingError(f"Body '{name}' not found.")

        self._placements[name] = Placement(
            x=x, y=y, z=z,
            axis_x=axis_x, axis_y=axis_y, axis_z=axis_z,
            angle_degrees=angle_degrees,
        )

        op = Operation(
            op_type=OpType.PLACE_BODY,
            params={"name": name, "x": x, "y": y, "z": z,
                     "axis_x": axis_x, "axis_y": axis_y, "axis_z": axis_z,
                     "angle_degrees": angle_degrees},
            description=f"Placed '{name}' at ({x},{y},{z})",
        )
        self._log.append(op)
        return {"body": name, "placement": self._placements[name].to_dict()}

    def add_mate(
        self,
        mate_type: str,
        body_a: str,
        entity_a: str,
        body_b: str,
        entity_b: str,
        **parameters: Any,
    ) -> dict[str, Any]:
        """Add a mate constraint between two bodies."""
        for name in [body_a, body_b]:
            if name not in self._bodies:
                raise ModelingError(f"Body '{name}' not found.")

        mate = MateConstraint(
            mate_type=mate_type,
            body_a=body_a, entity_a=entity_a,
            body_b=body_b, entity_b=entity_b,
            parameters=parameters,
        )
        self._mates.append(mate)

        op = Operation(
            op_type=OpType.ADD_MATE,
            params=mate.to_dict(),
            description=f"Mate {mate_type}: {body_a} ↔ {body_b}",
        )
        self._log.append(op)
        return {"mate_type": mate_type, "total_mates": len(self._mates)}

    def get_assembly_compound(self) -> TopoDS_Shape:
        """Build a compound shape from all placed bodies for export."""
        import cadquery as cq

        if not self._bodies:
            raise ModelingError("No bodies to assemble.")

        builder = TopoDS_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)

        for name, shape in self._bodies.items():
            placement = self._placements.get(name)
            if placement and (placement.x != 0 or placement.y != 0 or placement.z != 0
                              or placement.angle_degrees != 0):
                # Apply placement transform
                placed = kernel.translate(shape, placement.x, placement.y, placement.z)
                if placement.angle_degrees != 0:
                    placed = kernel.rotate(
                        placed,
                        axis=(placement.axis_x, placement.axis_y, placement.axis_z),
                        angle_degrees=placement.angle_degrees,
                    )
                builder.Add(compound, placed)
            else:
                builder.Add(compound, shape)

        return compound

    def export_assembly(self, path: str | Path) -> None:
        """Export the full assembly as a STEP file."""
        compound = self.get_assembly_compound()
        save_step(compound, path)

    # ------------------------------------------------------------------
    # INTERFERENCE DETECTION
    # ------------------------------------------------------------------

    def check_interference(self, body_a: str, body_b: str) -> dict[str, Any]:
        """Check if two bodies interfere (collide)."""
        if body_a not in self._bodies:
            raise ModelingError(f"Body '{body_a}' not found.")
        if body_b not in self._bodies:
            raise ModelingError(f"Body '{body_b}' not found.")

        shape_a = self._bodies[body_a]
        shape_b = self._bodies[body_b]

        # Apply placements before checking
        pa = self._placements.get(body_a)
        if pa and (pa.x or pa.y or pa.z or pa.angle_degrees):
            shape_a = kernel.translate(shape_a, pa.x, pa.y, pa.z)
            if pa.angle_degrees:
                shape_a = kernel.rotate(shape_a, (pa.axis_x, pa.axis_y, pa.axis_z), pa.angle_degrees)

        pb = self._placements.get(body_b)
        if pb and (pb.x or pb.y or pb.z or pb.angle_degrees):
            shape_b = kernel.translate(shape_b, pb.x, pb.y, pb.z)
            if pb.angle_degrees:
                shape_b = kernel.rotate(shape_b, (pb.axis_x, pb.axis_y, pb.axis_z), pb.angle_degrees)

        return kernel.check_interference(shape_a, shape_b)

    def check_all_interferences(self) -> dict[str, Any]:
        """Check all body pairs for interference."""
        names = list(self._bodies.keys())
        results = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                result = self.check_interference(names[i], names[j])
                result["body_a"] = names[i]
                result["body_b"] = names[j]
                results.append(result)

        interferences = [r for r in results if r["interferes"]]
        return {
            "pairs_checked": len(results),
            "interferences_found": len(interferences),
            "results": results,
        }

    # ------------------------------------------------------------------
    # BILL OF MATERIALS
    # ------------------------------------------------------------------

    def get_bom(self) -> dict[str, Any]:
        """Get bill of materials for all bodies."""
        from next3d.core.properties import MATERIALS, compute_physical_properties

        items = []
        total_mass = 0.0

        for name, shape in self._bodies.items():
            mat_key = self._body_materials.get(name, "steel")
            density = MATERIALS.get(mat_key, 0.00785)

            try:
                props = compute_physical_properties(shape, density)
                volume = props.volume
                mass = props.mass
            except Exception:
                volume = 0
                mass = 0

            items.append({
                "name": name,
                "material": mat_key,
                "volume_mm3": round(volume, 2),
                "mass_grams": round(mass, 2),
                "quantity": 1,
            })
            total_mass += mass

        return {
            "item_count": len(items),
            "total_mass_grams": round(total_mass, 2),
            "items": items,
        }

    # ------------------------------------------------------------------
    # DESIGN RULES CHECK
    # ------------------------------------------------------------------

    def check_design_rules(self, process: str = "cnc_milling") -> dict[str, Any]:
        """Check active body against manufacturing design rules."""
        from next3d.core.design_rules import check_design_rules
        g = self.graph
        result = check_design_rules(g, process)
        return result.to_dict()

    # ------------------------------------------------------------------
    # PARAMETRIC ENGINE
    # ------------------------------------------------------------------

    def set_parameter(self, name: str, value: float, description: str = "", unit: str = "mm") -> dict[str, Any]:
        """Define or update a named design parameter.

        Parameters can be referenced in operations using @name syntax.
        Changing a parameter triggers selective replay of affected operations.
        """
        self._parametric.define(name, value, description, unit)
        # Legacy compat
        self._parameters[name] = {"value": value, "description": description}
        return {
            "name": name,
            "value": value,
            "description": description,
            "unit": unit,
            "total_parameters": len(self._parametric.parameters),
        }

    def get_parameters(self) -> dict[str, Any]:
        """Get all named design parameters with dependency info."""
        deps = self._parametric.dependency_graph()
        return {
            "count": len(self._parametric.parameters),
            "parameters": {
                name: {
                    "value": p.value,
                    "description": p.description,
                    "unit": p.unit,
                    "used_by_operations": deps.get(name, []),
                }
                for name, p in self._parametric.parameters.items()
            },
        }

    def get_parameter(self, name: str) -> float:
        """Get a single parameter value. Raises if not found."""
        try:
            return self._parametric.get(name)
        except KeyError:
            raise ModelingError(
                f"Parameter '{name}' not found. "
                f"Available: {', '.join(self._parametric.parameters)}"
            )

    def update_parameter(self, name: str, new_value: float) -> dict[str, Any]:
        """Change a parameter and selectively replay affected operations.

        This is the core of parametric design: change one dimension,
        and only the operations that depend on it are re-executed.

        Returns:
            Dict with changed parameter, affected operations, and replay result.
        """
        if name not in self._parametric.parameters:
            raise ModelingError(f"Parameter '{name}' not defined.")

        old_value = self._parametric.get(name)
        affected_indices = self._parametric.change_parameter(name, new_value)

        # Legacy compat
        self._parameters[name] = {
            "value": new_value,
            "description": self._parametric.parameters[name].description,
        }

        if not affected_indices:
            return {
                "parameter": name,
                "old_value": old_value,
                "new_value": new_value,
                "affected_operations": 0,
                "message": "No operations depend on this parameter.",
            }

        # Get the replay plan with resolved values
        replay_plan = self._parametric.get_replay_plan([name])

        # Execute the replay: undo back to earliest affected, then replay
        body_name = self._active_body
        history = self._histories.get(body_name, [])

        # We need to undo to just before the earliest affected operation
        # Count how many ops to undo from current state
        total_ops = self._log.length
        earliest = affected_indices[0]
        ops_to_undo = total_ops - earliest

        # Undo affected operations
        for _ in range(min(ops_to_undo, len(history))):
            self._bodies[body_name] = history.pop()
            self._graphs[body_name] = None
            self._log.pop()

        # Replay with new parameter values
        replayed = 0
        for step in replay_plan:
            params = step["params"]
            op_type = step["op_type"]
            try:
                self._replay_operation(op_type, params)
                replayed += 1
            except Exception as e:
                return {
                    "parameter": name,
                    "old_value": old_value,
                    "new_value": new_value,
                    "affected_operations": len(affected_indices),
                    "replayed": replayed,
                    "error": f"Replay failed at op {step['op_index']}: {e}",
                }

        g = self.graph
        return {
            "parameter": name,
            "old_value": old_value,
            "new_value": new_value,
            "affected_operations": len(affected_indices),
            "replayed": replayed,
            "faces": len(g.faces),
            "features": len(g.features),
        }

    def _replay_operation(self, op_type: str, params: dict[str, Any]) -> None:
        """Replay a single operation with given parameters."""
        # Map op_type strings to session methods
        replay_map: dict[str, Callable] = {
            "create_box": lambda p: self.create_box(p["length"], p["width"], p["height"],
                                                     tuple(p.get("center", [0, 0, 0]))),
            "create_cylinder": lambda p: self.create_cylinder(p["radius"], p["height"],
                                                               tuple(p.get("center", [0, 0, 0])),
                                                               p.get("axis", "Z")),
            "create_sphere": lambda p: self.create_sphere(p["radius"],
                                                           tuple(p.get("center", [0, 0, 0]))),
            "add_hole": lambda p: self.add_hole(p["center_x"], p["center_y"], p["diameter"],
                                                 p.get("depth"), p.get("face_selector", ">Z")),
            "add_pocket": lambda p: self.add_pocket(p["center_x"], p["center_y"],
                                                     p["length"], p["width"], p["depth"],
                                                     p.get("face_selector", ">Z")),
            "add_boss": lambda p: self.add_boss(p["center_x"], p["center_y"],
                                                 p["diameter"], p["height"],
                                                 p.get("face_selector", ">Z")),
            "add_fillet": lambda p: self.add_fillet(p["radius"], p.get("edge_selector")),
            "add_chamfer": lambda p: self.add_chamfer(p["distance"], p.get("edge_selector")),
            "add_shell": lambda p: self.add_shell(p["thickness"], p.get("face_selector", ">Z")),
            "add_slot": lambda p: self.add_slot(p["center_x"], p["center_y"],
                                                 p["length"], p["width"], p["depth"],
                                                 p.get("angle", 0), p.get("face_selector", ">Z")),
        }

        handler = replay_map.get(op_type)
        if handler is None:
            raise ModelingError(f"Cannot replay operation type: {op_type}")
        handler(params)

    def get_dependency_graph(self) -> dict[str, Any]:
        """Get the parameter → operation dependency graph."""
        return self._parametric.dependency_graph()

    def design_table(
        self,
        param_ranges: dict[str, list[float]],
    ) -> dict[str, Any]:
        """Generate design variants from parameter combinations.

        Example:
            design_table({"wall_t": [2, 3, 4], "bolt_d": [4, 6]})
            → 6 variants, each with resolved operation parameters

        The AI can then replay each variant to generate separate STEP files.
        """
        try:
            variants = self._parametric.design_table(param_ranges)
        except KeyError as e:
            raise ModelingError(str(e))

        return {
            "variant_count": len(variants),
            "parameters_varied": list(param_ranges.keys()),
            "variants": variants,
        }

    def get_parametric_state(self) -> dict[str, Any]:
        """Get the full parametric state: parameters, bindings, dependencies."""
        return self._parametric.to_dict()

    # ------------------------------------------------------------------
    # GD&T ANNOTATIONS
    # ------------------------------------------------------------------

    def _get_gdt_set(self) -> Any:
        """Get or create the GDT annotation set for the active body."""
        from next3d.core.gdt import GDTAnnotationSet

        name = self._active_body
        if name not in self._gdt:
            self._gdt[name] = GDTAnnotationSet()
        return self._gdt[name]

    def add_datum(
        self,
        label: str,
        entity_id: str,
        description: str = "",
    ) -> dict[str, Any]:
        """Add a datum reference to the active body's GD&T annotations.

        Args:
            label: Datum label (e.g. "A", "B", "C").
            entity_id: persistent_id of the datum feature.
            description: Optional description.
        """
        from next3d.core.gdt import create_datum

        datum = create_datum(label, entity_id, description)
        gdt_set = self._get_gdt_set()
        gdt_set.datums.append(datum)

        op = Operation(
            op_type=OpType.ADD_DATUM,
            params={"label": label, "entity_id": entity_id, "description": description},
            description=f"Datum {label} on {entity_id}",
        )
        self._log.append(op)

        return {
            "label": label,
            "entity_id": entity_id,
            "total_datums": len(gdt_set.datums),
            "body": self._active_body,
        }

    def add_tolerance(
        self,
        tolerance_type: str,
        value: float,
        entity_id: str,
        datum_refs: list[str] | None = None,
        material_condition: str = "",
        description: str = "",
    ) -> dict[str, Any]:
        """Add a GD&T tolerance to the active body's annotations.

        Args:
            tolerance_type: One of the ToleranceType enum values.
            value: Tolerance value in mm.
            entity_id: persistent_id of the controlled feature.
            datum_refs: Datum labels this tolerance references.
            material_condition: "MMC", "LMC", "RFS", or "".
            description: Optional description.
        """
        from next3d.core.gdt import create_tolerance

        tol = create_tolerance(
            tolerance_type, value, entity_id,
            datum_refs=datum_refs,
            material_condition=material_condition,
            description=description,
        )
        gdt_set = self._get_gdt_set()
        gdt_set.tolerances.append(tol)

        op = Operation(
            op_type=OpType.ADD_TOLERANCE,
            params={
                "tolerance_type": tolerance_type, "value": value,
                "entity_id": entity_id, "datum_refs": datum_refs or [],
                "material_condition": material_condition,
                "description": description,
            },
            description=f"{tolerance_type} {value}mm on {entity_id}",
        )
        self._log.append(op)

        return {
            "tolerance_type": tolerance_type,
            "value": value,
            "entity_id": entity_id,
            "total_tolerances": len(gdt_set.tolerances),
            "body": self._active_body,
        }

    def get_gdt(self) -> dict[str, Any]:
        """Get all GD&T annotations for the active body."""
        from next3d.core.gdt import gdt_to_dict, validate_gdt

        gdt_set = self._get_gdt_set()
        result = gdt_to_dict(gdt_set)

        # Include validation issues
        graph = self.graph
        issues = validate_gdt(gdt_set, graph)
        result["validation_issues"] = issues
        result["body"] = self._active_body

        return result

    def suggest_gdt(self) -> dict[str, Any]:
        """Auto-suggest GD&T annotations based on current geometry."""
        from next3d.core.gdt import suggest_gdt, gdt_to_dict

        graph = self.graph
        suggestions = suggest_gdt(graph)
        result = gdt_to_dict(suggestions)
        result["body"] = self._active_body
        return result

    # ------------------------------------------------------------------
    # TOPOLOGY OPTIMIZATION
    # ------------------------------------------------------------------

    def add_load(
        self,
        name: str,
        fx: float,
        fy: float,
        fz: float,
        px: float,
        py: float,
        pz: float,
    ) -> dict[str, Any]:
        """Add a load case for topology optimization.

        Args:
            name: Load case name.
            fx, fy, fz: Force components in Newtons.
            px, py, pz: Application point coordinates.
        """
        from next3d.core.topology_opt import LoadCase

        load = LoadCase(name=name, force=(fx, fy, fz), application_point=(px, py, pz))
        self._loads.append(load)

        op = Operation(
            op_type=OpType.ADD_LOAD,
            params={"name": name, "fx": fx, "fy": fy, "fz": fz, "px": px, "py": py, "pz": pz},
            description=f"Load '{name}': ({fx},{fy},{fz})N at ({px},{py},{pz})",
        )
        self._log.append(op)

        return {
            "name": name,
            "force": {"x": fx, "y": fy, "z": fz},
            "application_point": {"x": px, "y": py, "z": pz},
            "total_loads": len(self._loads),
        }

    def add_boundary_condition(
        self,
        name: str,
        bc_type: str,
        face_selector: str,
    ) -> dict[str, Any]:
        """Add a boundary condition for topology optimization.

        Args:
            name: Boundary condition name.
            bc_type: Type — "fixed", "pinned", or "roller".
            face_selector: CadQuery face selector (">Z", "<Z", etc.).
        """
        from next3d.core.topology_opt import BoundaryCondition

        valid_types = ("fixed", "pinned", "roller")
        if bc_type not in valid_types:
            raise ModelingError(
                f"Invalid BC type '{bc_type}'. Use: {', '.join(valid_types)}"
            )

        bc = BoundaryCondition(name=name, bc_type=bc_type, face_selector=face_selector)
        self._boundary_conditions.append(bc)

        op = Operation(
            op_type=OpType.ADD_BOUNDARY_CONDITION,
            params={"name": name, "bc_type": bc_type, "face_selector": face_selector},
            description=f"BC '{name}': {bc_type} on {face_selector}",
        )
        self._log.append(op)

        return {
            "name": name,
            "bc_type": bc_type,
            "face_selector": face_selector,
            "total_boundary_conditions": len(self._boundary_conditions),
        }

    def run_topology_optimization(
        self,
        volume_fraction: float = 0.3,
        resolution: int = 10,
    ) -> dict[str, Any]:
        """Run topology optimization on the active body.

        Args:
            volume_fraction: Target volume fraction (0.3 = keep 30%).
            resolution: Voxel grid resolution along longest axis.

        Returns:
            Dict with optimization results including volume reduction,
            removal regions, density field, and suggestions.
        """
        from next3d.core.topology_opt import setup_optimization, run_optimization

        shape = self._require_shape()

        if not self._loads:
            raise ModelingError("No loads defined. Use add_load() first.")
        if not self._boundary_conditions:
            raise ModelingError(
                "No boundary conditions defined. Use add_boundary_condition() first."
            )

        setup = setup_optimization(
            loads=self._loads,
            constraints=self._boundary_conditions,
            volume_fraction=volume_fraction,
        )

        result = run_optimization(shape, setup, resolution=resolution)

        op = Operation(
            op_type=OpType.RUN_TOPOLOGY_OPT,
            params={
                "volume_fraction": volume_fraction,
                "resolution": resolution,
            },
            description=f"Topology opt: {result.volume_reduction_pct}% reduction",
        )
        self._log.append(op)

        return {
            "original_volume": result.original_volume,
            "optimized_volume": result.optimized_volume,
            "volume_reduction_pct": result.volume_reduction_pct,
            "removal_regions_count": len(result.removal_regions),
            "total_voxels": len(result.density_field),
            "kept_voxels": len(result.density_field) - len(result.removal_regions),
            "removal_regions": result.removal_regions,
            "density_field": result.density_field,
            "suggestions": result.suggestions,
            "body": self._active_body,
        }

    # ------------------------------------------------------------------
    # FEA — Finite Element Analysis
    # ------------------------------------------------------------------

    def run_fea(
        self,
        plate_width: float,
        plate_height: float,
        plate_thickness: float,
        grid_spacing_x: float = 0,
        grid_spacing_y: float = 0,
        rhs_size: str | None = None,
        material: str = "steel_mild",
        pressure_mpa: float = 0,
        point_loads: list[dict] | None = None,
        bc_type: str = "fixed_edges",
        weld_type: str = "full",
        weld_spacing: float = 50.0,
    ) -> dict[str, Any]:
        """Run plate/beam FEA on a stiffened panel.

        General-purpose structural analysis for any plate + stiffener configuration.
        """
        from next3d.core.fea import (
            FEASetup, RHS_SIZES, MATERIALS as FEA_MATERIALS, run_fea,
        )

        mat = FEA_MATERIALS.get(material)
        if mat is None:
            raise ModelingError(
                f"Unknown material: {material}. "
                f"Available: {', '.join(FEA_MATERIALS)}"
            )

        rhs = None
        if rhs_size:
            rhs = RHS_SIZES.get(rhs_size)
            if rhs is None:
                raise ModelingError(
                    f"Unknown RHS size: {rhs_size}. "
                    f"Available: {', '.join(RHS_SIZES)}"
                )

        setup = FEASetup(
            plate_width=plate_width,
            plate_height=plate_height,
            plate_thickness=plate_thickness,
            grid_spacing_x=grid_spacing_x or plate_width,
            grid_spacing_y=grid_spacing_y or plate_height,
            rhs_section=rhs,
            material=mat,
            pressure=pressure_mpa,
            point_loads=point_loads or [],
            bc_type=bc_type,
            weld_type=weld_type,
            weld_spacing=weld_spacing,
        )

        result = run_fea(setup)
        return result.to_dict()

    def run_fea_parametric(
        self,
        base_config: dict[str, Any],
        variations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run FEA across multiple configurations for comparison."""
        from next3d.core.fea import (
            FEASetup, RHS_SIZES, MATERIALS as FEA_MATERIALS,
            parametric_study,
        )
        import dataclasses

        mat = FEA_MATERIALS.get(base_config.get("material", "steel_mild"))
        rhs_key = base_config.get("rhs_size")
        rhs = RHS_SIZES.get(rhs_key) if rhs_key else None

        base = FEASetup(
            plate_width=base_config.get("plate_width", 300),
            plate_height=base_config.get("plate_height", 300),
            plate_thickness=base_config.get("plate_thickness", 3),
            grid_spacing_x=base_config.get("grid_spacing_x", 300),
            grid_spacing_y=base_config.get("grid_spacing_y", 300),
            rhs_section=rhs,
            material=mat,
            pressure=base_config.get("pressure_mpa", 0.1),
            bc_type=base_config.get("bc_type", "fixed_edges"),
            weld_type=base_config.get("weld_type", "full"),
        )

        # Process variations — resolve rhs_size strings to objects
        processed = []
        for var in variations:
            v = dict(var)
            if "rhs_size" in v:
                v["rhs_section"] = RHS_SIZES.get(v.pop("rhs_size"))
            if "material" in v:
                v["material"] = FEA_MATERIALS.get(v.pop("material"), mat)
            processed.append(v)

        results = parametric_study(base, processed)
        return {"config_count": len(results), "results": results}

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
        """Export current active body to a STEP file."""
        shape = self._require_shape()
        save_step(shape, path)

    def export_stl(
        self,
        path: str | Path,
        linear_deflection: float = 0.1,
        angular_deflection: float = 0.5,
    ) -> None:
        """Export current active body as STL."""
        kernel.export_stl(self._require_shape(), str(path), linear_deflection, angular_deflection)

    def export_3mf(
        self,
        path: str | Path,
        linear_deflection: float = 0.1,
        angular_deflection: float = 0.5,
    ) -> None:
        """Export current active body as 3MF."""
        kernel.export_3mf(self._require_shape(), str(path), linear_deflection, angular_deflection)

    def render_png(
        self,
        path: str | Path,
        width: int = 800,
        height: int = 600,
    ) -> None:
        """Render current active body to PNG/SVG."""
        kernel.render_png(self._require_shape(), str(path), width, height)

    # ------------------------------------------------------------------
    # CREATE operations (backward compatible — use "default" body)
    # ------------------------------------------------------------------

    def create_box(
        self,
        length: float,
        width: float,
        height: float,
        center: tuple[float, float, float] = (0, 0, 0),
    ) -> dict[str, Any]:
        """Create a box. Replaces active body geometry."""
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

    def create_revolve(
        self,
        points: list[tuple[float, float]],
        angle_degrees: float = 360.0,
        axis_origin: tuple[float, float] = (0, 0),
        axis_direction: tuple[float, float] = (0, 1),
        center: tuple[float, float, float] = (0, 0, 0),
    ) -> dict[str, Any]:
        shape = kernel.create_revolve(points, angle_degrees, axis_origin, axis_direction, center)
        op = Operation(
            op_type=OpType.CREATE_REVOLVE,
            params={
                "points": points, "angle_degrees": angle_degrees,
                "axis_origin": list(axis_origin), "axis_direction": list(axis_direction),
                "center": list(center),
            },
            description=f"Revolve {len(points)}-pt profile, {angle_degrees}°",
        )
        return self._apply(shape, op)

    def create_sweep(
        self,
        profile_points: list[tuple[float, float]],
        path_points: list[tuple[float, float, float]],
        center: tuple[float, float, float] = (0, 0, 0),
    ) -> dict[str, Any]:
        shape = kernel.create_sweep(profile_points, path_points, center)
        op = Operation(
            op_type=OpType.CREATE_SWEEP,
            params={
                "profile_points": profile_points,
                "path_points": path_points,
                "center": list(center),
            },
            description=f"Sweep {len(profile_points)}-pt profile along {len(path_points)}-pt path",
        )
        return self._apply(shape, op)

    def create_loft(
        self,
        sections: list[list[tuple[float, float]]],
        heights: list[float],
        ruled: bool = False,
        center: tuple[float, float, float] = (0, 0, 0),
    ) -> dict[str, Any]:
        shape = kernel.create_loft(sections, heights, ruled, center)
        op = Operation(
            op_type=OpType.CREATE_LOFT,
            params={
                "sections": sections, "heights": heights,
                "ruled": ruled, "center": list(center),
            },
            description=f"Loft {len(sections)} sections",
        )
        return self._apply(shape, op)

    # ------------------------------------------------------------------
    # MODIFY operations
    # ------------------------------------------------------------------

    def add_hole(
        self,
        center_x: float,
        center_y: float,
        diameter: float,
        depth: float | None = None,
        face_selector: str = ">Z",
    ) -> dict[str, Any]:
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

    def add_shell(
        self,
        thickness: float,
        face_selector: str = ">Z",
    ) -> dict[str, Any]:
        shape = kernel.add_shell(self._require_shape(), thickness, face_selector)
        op = Operation(
            op_type=OpType.ADD_SHELL,
            params={"thickness": thickness, "face_selector": face_selector},
            description=f"Shell t={thickness}",
        )
        return self._apply(shape, op)

    def add_draft(
        self,
        angle_degrees: float,
        face_selector: str = "|Z",
        pull_direction: tuple[float, float, float] = (0, 0, 1),
        plane_selector: str = "<Z",
    ) -> dict[str, Any]:
        shape = kernel.add_draft(
            self._require_shape(), angle_degrees, face_selector,
            pull_direction, plane_selector,
        )
        op = Operation(
            op_type=OpType.ADD_DRAFT,
            params={
                "angle_degrees": angle_degrees, "face_selector": face_selector,
                "pull_direction": list(pull_direction), "plane_selector": plane_selector,
            },
            description=f"Draft {angle_degrees}°",
        )
        return self._apply(shape, op)

    # ------------------------------------------------------------------
    # SKETCH operations
    # ------------------------------------------------------------------

    def create_sketch(self, plane: str = "XY") -> dict[str, Any]:
        """Start a new 2D sketch on the given plane.

        Args:
            plane: Sketch plane — "XY", "XZ", or "YZ".
        """
        from next3d.core.sketch import Sketch
        self._active_sketch = Sketch(plane=plane)
        op = Operation(
            op_type=OpType.CREATE_SKETCH,
            params={"plane": plane},
            description=f"Sketch on {plane}",
        )
        self._log.append(op)
        return {"plane": plane, "status": "sketch_active"}

    def sketch_add_line(
        self, x1: float, y1: float, x2: float, y2: float,
    ) -> dict[str, Any]:
        """Add a line to the active sketch."""
        sketch = self._require_sketch()
        eid = sketch.add_line(x1, y1, x2, y2)
        op = Operation(
            op_type=OpType.SKETCH_ADD_LINE,
            params={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            description=f"Line ({x1},{y1})->({x2},{y2})",
        )
        self._log.append(op)
        return {"entity_id": eid, "entity_count": len(sketch.entities)}

    def sketch_add_arc(
        self,
        cx: float,
        cy: float,
        radius: float,
        start_angle: float,
        end_angle: float,
    ) -> dict[str, Any]:
        """Add an arc to the active sketch."""
        sketch = self._require_sketch()
        eid = sketch.add_arc(cx, cy, radius, start_angle, end_angle)
        op = Operation(
            op_type=OpType.SKETCH_ADD_ARC,
            params={"cx": cx, "cy": cy, "radius": radius,
                    "start_angle": start_angle, "end_angle": end_angle},
            description=f"Arc center=({cx},{cy}) r={radius}",
        )
        self._log.append(op)
        return {"entity_id": eid, "entity_count": len(sketch.entities)}

    def sketch_add_circle(
        self, cx: float, cy: float, radius: float,
    ) -> dict[str, Any]:
        """Add a circle to the active sketch."""
        sketch = self._require_sketch()
        eid = sketch.add_circle(cx, cy, radius)
        op = Operation(
            op_type=OpType.SKETCH_ADD_CIRCLE,
            params={"cx": cx, "cy": cy, "radius": radius},
            description=f"Circle center=({cx},{cy}) r={radius}",
        )
        self._log.append(op)
        return {"entity_id": eid, "entity_count": len(sketch.entities)}

    def sketch_add_rect(
        self, cx: float, cy: float, width: float, height: float,
    ) -> dict[str, Any]:
        """Add a rectangle to the active sketch."""
        sketch = self._require_sketch()
        eid = sketch.add_rect(cx, cy, width, height)
        op = Operation(
            op_type=OpType.SKETCH_ADD_RECT,
            params={"cx": cx, "cy": cy, "width": width, "height": height},
            description=f"Rect center=({cx},{cy}) {width}x{height}",
        )
        self._log.append(op)
        return {"entity_id": eid, "entity_count": len(sketch.entities)}

    def sketch_add_constraint(
        self,
        constraint_type: str,
        entity_a: str,
        entity_b: str | None = None,
        value: float | None = None,
    ) -> dict[str, Any]:
        """Add a constraint to the active sketch."""
        sketch = self._require_sketch()
        cid = sketch.add_constraint(constraint_type, entity_a, entity_b, value)
        op = Operation(
            op_type=OpType.SKETCH_ADD_CONSTRAINT,
            params={"constraint_type": constraint_type, "entity_a": entity_a,
                    "entity_b": entity_b, "value": value},
            description=f"Constraint {constraint_type}",
        )
        self._log.append(op)
        return {"constraint_id": cid, "constraint_count": len(sketch.constraints)}

    def sketch_extrude(self, height: float) -> dict[str, Any]:
        """Extrude the active sketch to create/replace the active body."""
        sketch = self._require_sketch()
        shape = sketch.extrude(height)
        op = Operation(
            op_type=OpType.SKETCH_EXTRUDE,
            params={"height": height},
            description=f"Sketch extrude h={height}",
        )
        self._active_sketch = None
        return self._apply(shape, op)

    def sketch_revolve(
        self,
        angle_degrees: float = 360.0,
        axis_origin: tuple[float, float] = (0, 0),
        axis_direction: tuple[float, float] = (0, 1),
    ) -> dict[str, Any]:
        """Revolve the active sketch to create/replace the active body."""
        sketch = self._require_sketch()
        shape = sketch.revolve(angle_degrees, axis_origin, axis_direction)
        op = Operation(
            op_type=OpType.SKETCH_REVOLVE,
            params={"angle_degrees": angle_degrees,
                    "axis_origin": list(axis_origin),
                    "axis_direction": list(axis_direction)},
            description=f"Sketch revolve {angle_degrees}°",
        )
        self._active_sketch = None
        return self._apply(shape, op)

    def _require_sketch(self):
        """Return the active sketch or raise."""
        sketch = getattr(self, "_active_sketch", None)
        if sketch is None:
            raise ModelingError("No active sketch. Call create_sketch first.")
        return sketch

    # ------------------------------------------------------------------
    # BOOLEAN operations
    # ------------------------------------------------------------------

    def boolean_union(self, other_session: ModelingSession) -> dict[str, Any]:
        shape = kernel.boolean_union(self._require_shape(), other_session._require_shape())
        op = Operation(op_type=OpType.BOOLEAN_UNION, params={}, description="Boolean union")
        return self._apply(shape, op)

    def boolean_cut(self, tool_shape: TopoDS_Shape) -> dict[str, Any]:
        shape = kernel.boolean_cut(self._require_shape(), tool_shape)
        op = Operation(op_type=OpType.BOOLEAN_CUT, params={}, description="Boolean cut")
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
        """Undo the last operation on the active body."""
        name = self._active_body
        history = self._histories.get(name, [])
        if not history:
            raise ModelingError("Nothing to undo")
        self._bodies[name] = history.pop()
        self._graphs[name] = None
        removed = self._log.pop()
        g = self.graph
        return {
            "undone": removed.description if removed else "",
            "body": name,
            "faces": len(g.faces),
            "features": len(g.features),
        }

    # ------------------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------------------

    def to_script(self) -> str:
        return self._log.to_cadquery_script()

    def summary(self) -> dict[str, Any]:
        g = self.graph
        result = {
            "operations": self._log.length,
            "faces": len(g.faces),
            "edges": len(g.edges),
            "vertices": len(g.vertices),
            "features": len(g.features),
            "solids": len(g.solids),
            "feature_types": [f.feature_type for f in g.features],
        }
        if len(self._bodies) > 1:
            result["active_body"] = self._active_body
            result["total_bodies"] = len(self._bodies)
            result["body_names"] = list(self._bodies.keys())
        return result
