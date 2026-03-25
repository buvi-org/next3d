"""Tool executor — dispatches AI tool calls to the modeling session.

This is the runtime that connects tool schemas to actual geometry operations.
The AI sends a tool name + parameters, the executor validates, runs, and
returns a structured result.
"""

from __future__ import annotations

import math
from typing import Any

from next3d.core.schema import SurfaceType
from next3d.graph.query import execute_query
from next3d.modeling import kernel
from next3d.modeling.session import ModelingSession
from next3d.tools.schema import TOOL_SCHEMAS


class ToolResult:
    """Structured result from a tool execution."""

    __slots__ = ("success", "message", "data")

    def __init__(
        self,
        success: bool = True,
        message: str = "",
        data: dict[str, Any] | None = None,
    ):
        self.success = success
        self.message = message
        self.data = data or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
        }


class ToolExecutor:
    """Executes tool calls against a modeling session.

    Usage:
        executor = ToolExecutor()
        result = executor.call("create_box", {"length": 100, "width": 60, "height": 20})
        result = executor.call("add_hole", {"center_x": 0, "center_y": 0, "diameter": 10})
        result = executor.call("get_summary", {})
    """

    def __init__(self, session: ModelingSession | None = None):
        self._session = session or ModelingSession()

    @property
    def session(self) -> ModelingSession:
        return self._session

    def call(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """Execute a tool call.

        Args:
            tool_name: Name of the tool (must be in TOOL_SCHEMAS).
            params: Tool parameters (will be validated against the schema).

        Returns:
            ToolResult with success/failure, message, and data.
        """
        if tool_name not in TOOL_SCHEMAS:
            return ToolResult(
                success=False,
                message=f"Unknown tool: {tool_name}. Available: {', '.join(sorted(TOOL_SCHEMAS))}",
            )

        # Validate parameters
        schema_cls = TOOL_SCHEMAS[tool_name]
        try:
            validated = schema_cls(**params)
        except Exception as e:
            return ToolResult(success=False, message=f"Invalid parameters: {e}")

        # Dispatch to handler
        try:
            handler = getattr(self, f"_handle_{tool_name}", None)
            if handler is None:
                return ToolResult(success=False, message=f"Tool '{tool_name}' not implemented")
            return handler(validated)
        except Exception as e:
            return ToolResult(success=False, message=f"Operation failed: {e}")

    # ------------------------------------------------------------------
    # CREATE handlers
    # ------------------------------------------------------------------

    def _handle_create_box(self, p) -> ToolResult:
        data = self._session.create_box(p.length, p.width, p.height, (p.center_x, p.center_y, p.center_z))
        return ToolResult(message=f"Created box {p.length}×{p.width}×{p.height}", data=data)

    def _handle_create_cylinder(self, p) -> ToolResult:
        data = self._session.create_cylinder(p.radius, p.height, (p.center_x, p.center_y, p.center_z), p.axis)
        return ToolResult(message=f"Created cylinder r={p.radius} h={p.height}", data=data)

    def _handle_create_sphere(self, p) -> ToolResult:
        data = self._session.create_sphere(p.radius, (p.center_x, p.center_y, p.center_z))
        return ToolResult(message=f"Created sphere r={p.radius}", data=data)

    def _handle_create_extrusion(self, p) -> ToolResult:
        points = [tuple(pt) for pt in p.points]
        data = self._session.create_extrusion(points, p.height, (p.center_x, p.center_y, p.center_z))
        return ToolResult(message=f"Created extrusion, {len(points)} vertices", data=data)

    def _handle_create_revolve(self, p) -> ToolResult:
        points = [tuple(pt) for pt in p.points]
        data = self._session.create_revolve(
            points, p.angle_degrees,
            (p.axis_origin_x, p.axis_origin_z),
            (p.axis_direction_x, p.axis_direction_z),
            (p.center_x, p.center_y, p.center_z),
        )
        return ToolResult(message=f"Created revolve, {len(points)} profile points, {p.angle_degrees}°", data=data)

    def _handle_create_sweep(self, p) -> ToolResult:
        profile = [tuple(pt) for pt in p.profile_points]
        path = [tuple(pt) for pt in p.path_points]
        data = self._session.create_sweep(profile, path, (p.center_x, p.center_y, p.center_z))
        return ToolResult(message=f"Created sweep along {len(path)}-point path", data=data)

    def _handle_create_loft(self, p) -> ToolResult:
        sections = [[tuple(pt) for pt in sec] for sec in p.sections]
        data = self._session.create_loft(
            sections, p.heights, p.ruled, (p.center_x, p.center_y, p.center_z),
        )
        return ToolResult(message=f"Created loft with {len(sections)} sections", data=data)

    # ------------------------------------------------------------------
    # MODIFY handlers
    # ------------------------------------------------------------------

    def _handle_add_hole(self, p) -> ToolResult:
        data = self._session.add_hole(p.center_x, p.center_y, p.diameter, p.depth, p.face_selector)
        return ToolResult(message=f"Added hole ⌀{p.diameter}", data=data)

    def _handle_add_counterbore_hole(self, p) -> ToolResult:
        data = self._session.add_counterbore_hole(
            p.center_x, p.center_y, p.hole_diameter, p.cb_diameter, p.cb_depth, p.depth, p.face_selector,
        )
        return ToolResult(message=f"Added counterbore hole ⌀{p.hole_diameter}/⌀{p.cb_diameter}", data=data)

    def _handle_add_pocket(self, p) -> ToolResult:
        data = self._session.add_pocket(p.center_x, p.center_y, p.length, p.width, p.depth, p.face_selector)
        return ToolResult(message=f"Added pocket {p.length}×{p.width}×{p.depth}", data=data)

    def _handle_add_circular_pocket(self, p) -> ToolResult:
        data = self._session.add_circular_pocket(p.center_x, p.center_y, p.diameter, p.depth, p.face_selector)
        return ToolResult(message=f"Added circular pocket ⌀{p.diameter}", data=data)

    def _handle_add_boss(self, p) -> ToolResult:
        data = self._session.add_boss(p.center_x, p.center_y, p.diameter, p.height, p.face_selector)
        return ToolResult(message=f"Added boss ⌀{p.diameter}×{p.height}", data=data)

    def _handle_add_slot(self, p) -> ToolResult:
        data = self._session.add_slot(
            p.center_x, p.center_y, p.length, p.width, p.depth, p.angle, p.face_selector,
        )
        return ToolResult(message=f"Added slot {p.length}×{p.width}×{p.depth}", data=data)

    def _handle_add_fillet(self, p) -> ToolResult:
        data = self._session.add_fillet(p.radius, p.edge_selector)
        return ToolResult(message=f"Added fillet r={p.radius}", data=data)

    def _handle_add_chamfer(self, p) -> ToolResult:
        data = self._session.add_chamfer(p.distance, p.edge_selector)
        return ToolResult(message=f"Added chamfer d={p.distance}", data=data)

    def _handle_add_shell(self, p) -> ToolResult:
        data = self._session.add_shell(p.thickness, p.face_selector)
        return ToolResult(message=f"Shelled to t={p.thickness}", data=data)

    def _handle_add_draft(self, p) -> ToolResult:
        data = self._session.add_draft(
            p.angle_degrees, p.face_selector,
            (p.pull_direction_x, p.pull_direction_y, p.pull_direction_z),
            p.plane_selector,
        )
        return ToolResult(message=f"Added draft {p.angle_degrees}°", data=data)

    # ------------------------------------------------------------------
    # BOOLEAN handlers
    # ------------------------------------------------------------------

    def _handle_boolean_cut(self, p) -> ToolResult:
        creators = {
            "box": kernel.create_box,
            "cylinder": kernel.create_cylinder,
            "sphere": kernel.create_sphere,
        }
        creator = creators.get(p.tool_type)
        if not creator:
            return ToolResult(success=False, message=f"Unknown tool type: {p.tool_type}")
        tool_shape = creator(**p.tool_params)
        data = self._session.boolean_cut(tool_shape)
        return ToolResult(message=f"Boolean cut with {p.tool_type}", data=data)

    # ------------------------------------------------------------------
    # TRANSFORM handlers
    # ------------------------------------------------------------------

    def _handle_translate(self, p) -> ToolResult:
        data = self._session.translate(p.dx, p.dy, p.dz)
        return ToolResult(message=f"Translated ({p.dx},{p.dy},{p.dz})", data=data)

    def _handle_rotate(self, p) -> ToolResult:
        data = self._session.rotate(
            (p.axis_x, p.axis_y, p.axis_z), p.angle_degrees, (p.center_x, p.center_y, p.center_z),
        )
        return ToolResult(message=f"Rotated {p.angle_degrees}°", data=data)

    # ------------------------------------------------------------------
    # QUERY handlers
    # ------------------------------------------------------------------

    def _handle_query_geometry(self, p) -> ToolResult:
        graph = self._session.graph
        results = execute_query(graph, p.query)
        return ToolResult(
            message=f"Query returned {len(results)} results",
            data={"results": results, "count": len(results)},
        )

    def _handle_get_summary(self, _p) -> ToolResult:
        data = self._session.summary()
        return ToolResult(message="Session summary", data=data)

    def _handle_get_features(self, p) -> ToolResult:
        graph = self._session.graph
        features = graph.features
        if p.feature_type:
            features = [f for f in features if f.feature_type.value == p.feature_type]
        data = {
            "count": len(features),
            "features": [
                {
                    "id": f.persistent_id,
                    "type": f.feature_type.value,
                    "parameters": f.parameters,
                    "description": f.description,
                }
                for f in features
            ],
        }
        return ToolResult(message=f"Found {len(features)} features", data=data)

    def _handle_find_faces(self, p) -> ToolResult:
        graph = self._session.graph
        faces = graph.faces

        if p.surface_type:
            try:
                st = SurfaceType(p.surface_type)
                faces = [f for f in faces if f.surface_type == st]
            except ValueError:
                return ToolResult(success=False, message=f"Unknown surface type: {p.surface_type}")

        if p.min_radius is not None:
            faces = [f for f in faces if f.radius is not None and f.radius >= p.min_radius]
        if p.max_radius is not None:
            faces = [f for f in faces if f.radius is not None and f.radius <= p.max_radius]

        if any(v is not None for v in [p.normal_x, p.normal_y, p.normal_z]):
            nx = p.normal_x or 0
            ny = p.normal_y or 0
            nz = p.normal_z or 0
            mag = math.sqrt(nx * nx + ny * ny + nz * nz)
            if mag > 0:
                nx, ny, nz = nx / mag, ny / mag, nz / mag
                filtered = []
                for f in faces:
                    if f.normal:
                        dot = abs(f.normal.x * nx + f.normal.y * ny + f.normal.z * nz)
                        if dot > 0.95:
                            filtered.append(f)
                faces = filtered

        data = {
            "count": len(faces),
            "faces": [
                {
                    "id": f.persistent_id,
                    "surface_type": f.surface_type.value,
                    "area": round(f.area, 4),
                    "centroid": {"x": f.centroid.x, "y": f.centroid.y, "z": f.centroid.z},
                    "radius": f.radius,
                }
                for f in faces
            ],
        }
        return ToolResult(message=f"Found {len(faces)} faces", data=data)

    # ------------------------------------------------------------------
    # SESSION handlers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # DIMENSION & DRAWING handlers
    # ------------------------------------------------------------------

    def _handle_add_dimension(self, p) -> ToolResult:
        data = self._session.add_dimension(
            p.dim_type, p.value, p.entity_ids, p.label,
            p.tolerance_plus, p.tolerance_minus,
        )
        return ToolResult(message=f"Added {p.dim_type} dimension: {p.value}", data=data)

    def _handle_get_dimensions(self, _p) -> ToolResult:
        data = self._session.get_dimensions()
        return ToolResult(message=f"{data['count']} dimensions", data=data)

    def _handle_auto_dimension(self, _p) -> ToolResult:
        data = self._session.auto_dimension()
        return ToolResult(message=f"Auto-generated {data['count']} dimensions", data=data)

    def _handle_export_drawing(self, p) -> ToolResult:
        self._session.export_drawing(
            p.output_path, p.views, p.title, p.show_hidden,
            p.page_width, p.page_height,
        )
        return ToolResult(message=f"Drawing exported to {p.output_path}")

    def _handle_export_section_drawing(self, p) -> ToolResult:
        self._session.export_section_drawing(
            p.output_path, p.section_plane, p.section_offset, p.title,
        )
        return ToolResult(message=f"Section drawing exported to {p.output_path}")

    def _handle_export_dxf(self, p) -> ToolResult:
        self._session.export_dxf(
            p.output_path, (p.projection_dir_x, p.projection_dir_y, p.projection_dir_z),
        )
        return ToolResult(message=f"DXF exported to {p.output_path}")

    # ------------------------------------------------------------------
    # DESIGN INTELLIGENCE handlers
    # ------------------------------------------------------------------

    def _handle_check_design_rules(self, p) -> ToolResult:
        data = self._session.check_design_rules(p.process)
        status = "PASSED" if data["passed"] else f"FAILED ({data['errors']} errors, {data['warnings']} warnings)"
        return ToolResult(message=f"Design check [{p.process}]: {status}", data=data)

    def _handle_set_parameter(self, p) -> ToolResult:
        data = self._session.set_parameter(p.name, p.value, p.description)
        return ToolResult(message=f"Parameter '{p.name}' = {p.value}", data=data)

    def _handle_get_parameters(self, _p) -> ToolResult:
        data = self._session.get_parameters()
        return ToolResult(message=f"{data['count']} parameters", data=data)

    def _handle_update_parameter(self, p) -> ToolResult:
        data = self._session.update_parameter(p.name, p.new_value)
        replayed = data.get("replayed", 0)
        return ToolResult(
            message=f"Updated {p.name} → {p.new_value} ({replayed} operations replayed)",
            data=data,
        )

    def _handle_design_table(self, p) -> ToolResult:
        data = self._session.design_table(p.param_ranges)
        return ToolResult(
            message=f"Design table: {data['variant_count']} variants",
            data=data,
        )

    def _handle_get_parametric_state(self, _p) -> ToolResult:
        data = self._session.get_parametric_state()
        return ToolResult(message="Parametric state", data=data)

    # ------------------------------------------------------------------
    # MULTI-BODY handlers
    # ------------------------------------------------------------------

    def _handle_create_named_body(self, p) -> ToolResult:
        params = {}
        if p.length is not None:
            params["length"] = p.length
        if p.width is not None:
            params["width"] = p.width
        if p.height is not None:
            params["height"] = p.height
        if p.radius is not None:
            params["radius"] = p.radius
        data = self._session.create_body(p.name, p.shape_type, p.material, **params)
        return ToolResult(message=f"Created body '{p.name}' ({p.shape_type})", data=data)

    def _handle_set_active_body(self, p) -> ToolResult:
        data = self._session.set_active_body(p.name)
        return ToolResult(message=f"Active body: {p.name}", data=data)

    def _handle_list_bodies(self, _p) -> ToolResult:
        data = self._session.list_bodies()
        return ToolResult(message=f"{data['count']} bodies", data=data)

    def _handle_delete_body(self, p) -> ToolResult:
        data = self._session.delete_body(p.name)
        return ToolResult(message=f"Deleted body '{p.name}'", data=data)

    def _handle_place_body(self, p) -> ToolResult:
        data = self._session.place_body(
            p.name, p.x, p.y, p.z, p.axis_x, p.axis_y, p.axis_z, p.angle_degrees,
        )
        return ToolResult(message=f"Placed '{p.name}' at ({p.x},{p.y},{p.z})", data=data)

    def _handle_check_interference(self, p) -> ToolResult:
        data = self._session.check_interference(p.body_a, p.body_b)
        status = "INTERFERES" if data["interferes"] else "clear"
        return ToolResult(message=f"{p.body_a} ↔ {p.body_b}: {status}", data=data)

    def _handle_get_bom(self, _p) -> ToolResult:
        data = self._session.get_bom()
        return ToolResult(message=f"BOM: {data['item_count']} items, {data['total_mass_grams']}g", data=data)

    def _handle_add_standard_part(self, p) -> ToolResult:
        from next3d.parts.fasteners import (
            iso_hex_bolt, iso_hex_nut, iso_flat_washer, iso_socket_head_cap_screw,
        )

        generators = {
            "hex_bolt": lambda: iso_hex_bolt(p.size, p.length or 20),
            "hex_nut": lambda: iso_hex_nut(p.size),
            "flat_washer": lambda: iso_flat_washer(p.size),
            "socket_head_cap_screw": lambda: iso_socket_head_cap_screw(p.size, p.length or 20),
        }
        gen = generators.get(p.part_type)
        if gen is None:
            return ToolResult(success=False, message=f"Unknown part type: {p.part_type}")

        shape = gen()
        # Add as a named body
        session = self._session
        if p.name in session._bodies:
            return ToolResult(success=False, message=f"Body '{p.name}' already exists.")

        session._bodies[p.name] = shape
        session._body_materials[p.name] = "steel"
        session._graphs[p.name] = None
        old_active = session._active_body
        session._active_body = p.name
        g = session.graph
        session._active_body = old_active

        return ToolResult(
            message=f"Added {p.part_type} {p.size} as '{p.name}'",
            data={"name": p.name, "part_type": p.part_type, "size": p.size,
                  "faces": len(g.faces), "total_bodies": len(session._bodies)},
        )

    def _handle_export_assembly(self, p) -> ToolResult:
        self._session.export_assembly(p.output_path)
        return ToolResult(message=f"Exported assembly to {p.output_path}")

    def _handle_add_mate_constraint(self, p) -> ToolResult:
        params = {}
        if p.distance is not None:
            params["distance"] = p.distance
        if p.angle is not None:
            params["angle"] = p.angle
        data = self._session.add_mate(
            p.mate_type, p.body_a, p.entity_a, p.body_b, p.entity_b, **params,
        )
        return ToolResult(message=f"Added {p.mate_type} mate: {p.body_a} ↔ {p.body_b}", data=data)

    # ------------------------------------------------------------------
    # SKETCH handlers
    # ------------------------------------------------------------------

    def _handle_create_sketch(self, p) -> ToolResult:
        data = self._session.create_sketch(p.plane)
        return ToolResult(message=f"Sketch started on {p.plane}", data=data)

    def _handle_sketch_add_line(self, p) -> ToolResult:
        data = self._session.sketch_add_line(p.x1, p.y1, p.x2, p.y2)
        return ToolResult(message=f"Added line ({p.x1},{p.y1})->({p.x2},{p.y2})", data=data)

    def _handle_sketch_add_arc(self, p) -> ToolResult:
        data = self._session.sketch_add_arc(p.cx, p.cy, p.radius, p.start_angle, p.end_angle)
        return ToolResult(message=f"Added arc r={p.radius}", data=data)

    def _handle_sketch_add_circle(self, p) -> ToolResult:
        data = self._session.sketch_add_circle(p.cx, p.cy, p.radius)
        return ToolResult(message=f"Added circle r={p.radius}", data=data)

    def _handle_sketch_add_rect(self, p) -> ToolResult:
        data = self._session.sketch_add_rect(p.cx, p.cy, p.width, p.height)
        return ToolResult(message=f"Added rect {p.width}x{p.height}", data=data)

    def _handle_sketch_add_constraint(self, p) -> ToolResult:
        data = self._session.sketch_add_constraint(
            p.constraint_type, p.entity_a, p.entity_b, p.value,
        )
        return ToolResult(message=f"Added {p.constraint_type} constraint", data=data)

    def _handle_sketch_extrude(self, p) -> ToolResult:
        data = self._session.sketch_extrude(p.height)
        return ToolResult(message=f"Extruded sketch h={p.height}", data=data)

    def _handle_sketch_revolve(self, p) -> ToolResult:
        data = self._session.sketch_revolve(
            p.angle_degrees,
            (p.axis_origin_x, p.axis_origin_y),
            (p.axis_direction_x, p.axis_direction_y),
        )
        return ToolResult(message=f"Revolved sketch {p.angle_degrees} deg", data=data)

    # ------------------------------------------------------------------
    # GD&T handlers
    # ------------------------------------------------------------------

    def _handle_add_datum(self, p) -> ToolResult:
        data = self._session.add_datum(p.label, p.entity_id, p.description)
        return ToolResult(message=f"Added datum {p.label}", data=data)

    def _handle_add_tolerance(self, p) -> ToolResult:
        data = self._session.add_tolerance(
            p.tolerance_type, p.value, p.entity_id,
            datum_refs=p.datum_refs,
            material_condition=p.material_condition,
            description=p.description,
        )
        return ToolResult(message=f"Added {p.tolerance_type} tolerance ({p.value}mm)", data=data)

    def _handle_get_gdt(self, _p) -> ToolResult:
        data = self._session.get_gdt()
        return ToolResult(
            message=f"GD&T: {data['datum_count']} datums, {data['tolerance_count']} tolerances",
            data=data,
        )

    def _handle_suggest_gdt(self, _p) -> ToolResult:
        data = self._session.suggest_gdt()
        return ToolResult(
            message=f"Suggested {data['datum_count']} datums, {data['tolerance_count']} tolerances",
            data=data,
        )

    # ------------------------------------------------------------------
    # TOPOLOGY OPTIMIZATION handlers
    # ------------------------------------------------------------------

    def _handle_add_load(self, p) -> ToolResult:
        data = self._session.add_load(p.name, p.fx, p.fy, p.fz, p.px, p.py, p.pz)
        mag = (p.fx ** 2 + p.fy ** 2 + p.fz ** 2) ** 0.5
        return ToolResult(message=f"Added load '{p.name}' ({mag:.0f}N)", data=data)

    def _handle_add_boundary_condition(self, p) -> ToolResult:
        data = self._session.add_boundary_condition(p.name, p.bc_type, p.face_selector)
        return ToolResult(message=f"Added BC '{p.name}' ({p.bc_type} on {p.face_selector})", data=data)

    def _handle_run_topology_optimization(self, p) -> ToolResult:
        data = self._session.run_topology_optimization(p.volume_fraction, p.resolution)
        return ToolResult(
            message=f"Topology optimization: {data['volume_reduction_pct']}% volume reduction",
            data=data,
        )

    # ------------------------------------------------------------------
    # FEA handlers
    # ------------------------------------------------------------------

    def _handle_run_fea(self, p) -> ToolResult:
        data = self._session.run_fea(
            plate_width=p.plate_width,
            plate_height=p.plate_height,
            plate_thickness=p.plate_thickness,
            grid_spacing_x=p.grid_spacing_x,
            grid_spacing_y=p.grid_spacing_y,
            rhs_size=p.rhs_size,
            material=p.material,
            pressure_mpa=p.pressure_mpa,
            point_loads=p.point_loads,
            bc_type=p.bc_type,
            weld_type=p.weld_type,
            weld_spacing=p.weld_spacing,
        )
        verdict = data.get("verdict", "")
        return ToolResult(message=f"FEA: {verdict}", data=data)

    def _handle_run_fea_parametric(self, p) -> ToolResult:
        data = self._session.run_fea_parametric(p.base_config, p.variations)
        return ToolResult(
            message=f"FEA parametric: {data['config_count']} configurations compared",
            data=data,
        )

    # ------------------------------------------------------------------
    # SESSION handlers
    # ------------------------------------------------------------------

    def _handle_undo(self, _p) -> ToolResult:
        data = self._session.undo()
        return ToolResult(message=f"Undone: {data.get('undone', '')}", data=data)

    def _handle_export_step(self, p) -> ToolResult:
        self._session.export_step(p.output_path)
        return ToolResult(message=f"Exported STEP to {p.output_path}")

    def _handle_export_stl(self, p) -> ToolResult:
        self._session.export_stl(p.output_path, p.linear_deflection, p.angular_deflection)
        return ToolResult(message=f"Exported STL to {p.output_path}")

    def _handle_export_3mf(self, p) -> ToolResult:
        self._session.export_3mf(p.output_path, p.linear_deflection, p.angular_deflection)
        return ToolResult(message=f"Exported 3MF to {p.output_path}")

    def _handle_render_png(self, p) -> ToolResult:
        self._session.render_png(p.output_path, p.width, p.height)
        return ToolResult(message=f"Rendered to {p.output_path}")

    def _handle_export_script(self, _p) -> ToolResult:
        script = self._session.to_script()
        return ToolResult(message="CadQuery script generated", data={"script": script})

    def _handle_load_step(self, p) -> ToolResult:
        data = self._session.load_step(p.path)
        return ToolResult(message=f"Loaded {p.path}", data=data)
