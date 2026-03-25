"""Tests for AI modeling pipeline: create → query → modify → export."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from next3d.cli.main import cli
from next3d.modeling.kernel import (
    create_box, create_cylinder, create_sphere, create_extrusion,
    create_revolve, create_sweep, create_loft,
    add_hole, add_pocket, add_boss, add_fillet, add_chamfer, add_slot,
    add_shell, add_draft, export_stl,
    boolean_union, boolean_cut, translate, rotate,
)
from next3d.modeling.session import ModelingSession, ModelingError
from next3d.modeling.operations import Operation, OperationLog, OpType
from next3d.tools.executor import ToolExecutor
from next3d.tools.formats import to_openai_tools, to_mcp_tools, to_anthropic_tools
from next3d.tools.schema import TOOL_SCHEMAS
from next3d.graph.semantic import build_semantic_graph_from_shape
from next3d.core.brep import save_step

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------------

class TestKernelCreate:
    def test_create_box(self):
        shape = create_box(100, 60, 20)
        g = build_semantic_graph_from_shape(shape)
        assert len(g.faces) == 6
        assert len(g.solids) == 1

    def test_create_cylinder(self):
        shape = create_cylinder(10, 50)
        g = build_semantic_graph_from_shape(shape)
        assert len(g.solids) == 1
        cylinders = [f for f in g.faces if f.surface_type.value == "cylinder"]
        assert len(cylinders) >= 1

    def test_create_sphere(self):
        shape = create_sphere(25)
        g = build_semantic_graph_from_shape(shape)
        assert len(g.solids) == 1

    def test_create_extrusion(self):
        # L-shaped profile
        points = [(0, 0), (40, 0), (40, 10), (10, 10), (10, 30), (0, 30)]
        shape = create_extrusion(points, 15)
        g = build_semantic_graph_from_shape(shape)
        assert len(g.solids) == 1
        assert len(g.faces) > 6  # more than a box


class TestKernelModify:
    def test_add_hole(self):
        box = create_box(100, 60, 20)
        drilled = add_hole(box, 0, 0, 10)
        g = build_semantic_graph_from_shape(drilled)
        holes = [f for f in g.features if "hole" in f.feature_type.value]
        assert len(holes) >= 1

    def test_add_pocket(self):
        box = create_box(100, 60, 20)
        pocketed = add_pocket(box, 0, 0, 30, 20, 5)
        g = build_semantic_graph_from_shape(pocketed)
        assert len(g.faces) > 6

    def test_add_boss(self):
        box = create_box(100, 60, 20)
        bossed = add_boss(box, 0, 0, 20, 10)
        g = build_semantic_graph_from_shape(bossed)
        bosses = [f for f in g.features if f.feature_type.value == "boss"]
        assert len(bosses) >= 1

    def test_add_fillet(self):
        box = create_box(100, 60, 20)
        filleted = add_fillet(box, 2.0, "|Z")
        g = build_semantic_graph_from_shape(filleted)
        assert len(g.faces) > 6

    def test_add_chamfer(self):
        box = create_box(100, 60, 20)
        chamfered = add_chamfer(box, 2.0, "|Z")
        g = build_semantic_graph_from_shape(chamfered)
        assert len(g.faces) > 6

    def test_add_slot(self):
        box = create_box(100, 60, 20)
        slotted = add_slot(box, 0, 0, 40, 10, 5)
        g = build_semantic_graph_from_shape(slotted)
        assert len(g.faces) > 6

    def test_boolean_union(self):
        box = create_box(100, 60, 20)
        cyl = create_cylinder(10, 40, center=(0, 0, 10))
        merged = boolean_union(box, cyl)
        g = build_semantic_graph_from_shape(merged)
        assert len(g.faces) > 6

    def test_boolean_cut(self):
        box = create_box(100, 60, 20)
        cyl = create_cylinder(10, 40, center=(0, 0, 0))
        cut = boolean_cut(box, cyl)
        g = build_semantic_graph_from_shape(cut)
        assert len(g.faces) > 6

    def test_translate(self):
        box = create_box(10, 10, 10)
        moved = translate(box, dx=50)
        g = build_semantic_graph_from_shape(moved)
        # centroid should be near x=50
        assert g.solids[0].centroid.x > 40

    def test_rotate(self):
        box = create_box(100, 10, 10)
        rotated = rotate(box, axis=(0, 0, 1), angle_degrees=90)
        g = build_semantic_graph_from_shape(rotated)
        assert len(g.solids) == 1


# ---------------------------------------------------------------------------
# Phase 4 — Advanced modeling kernel tests
# ---------------------------------------------------------------------------

class TestKernelAdvancedCreate:
    def test_create_revolve_full(self):
        """Revolve a rectangular profile 360° to make a hollow cylinder / washer shape."""
        # Profile in XZ plane: a small rectangle to the right of the Z axis
        points = [(5, 0), (10, 0), (10, 20), (5, 20)]
        shape = create_revolve(points, 360.0, axis_origin=(0, 0), axis_direction=(0, 1))
        g = build_semantic_graph_from_shape(shape)
        assert len(g.solids) == 1
        # A revolved rectangle produces cylindrical and planar faces
        cylinders = [f for f in g.faces if f.surface_type.value == "cylinder"]
        assert len(cylinders) >= 2  # inner + outer

    def test_create_revolve_partial(self):
        """Revolve 180° to make a half-pipe."""
        points = [(5, 0), (10, 0), (10, 20), (5, 20)]
        shape = create_revolve(points, 180.0, axis_origin=(0, 0), axis_direction=(0, 1))
        g = build_semantic_graph_from_shape(shape)
        assert len(g.solids) == 1

    def test_create_loft_two_sections(self):
        """Loft between a square and a circle-approximating polygon."""
        import math
        square = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
        # Approximate circle with 8-gon at height 30
        r = 8
        octagon = [(r * math.cos(i * math.pi / 4), r * math.sin(i * math.pi / 4)) for i in range(8)]
        shape = create_loft([square, octagon], [0, 30])
        g = build_semantic_graph_from_shape(shape)
        assert len(g.solids) == 1
        assert len(g.faces) > 4  # more complex than a prism


class TestKernelAdvancedModify:
    def test_add_shell(self):
        """Shell a box to make an open-top container."""
        box = create_box(60, 40, 30)
        shelled = add_shell(box, 2.0, ">Z")
        g = build_semantic_graph_from_shape(shelled)
        # Shelling adds inner faces
        assert len(g.faces) > 6

    def test_add_shell_thick(self):
        """Shell with larger wall thickness."""
        box = create_box(100, 80, 50)
        shelled = add_shell(box, 5.0, ">Z")
        g = build_semantic_graph_from_shape(shelled)
        assert len(g.solids) == 1
        assert len(g.faces) > 6

    def test_export_stl(self, tmp_path):
        """Export a box as STL."""
        box = create_box(50, 30, 10)
        out = str(tmp_path / "test.stl")
        export_stl(box, out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0


class TestSessionAdvanced:
    def test_revolve_session(self):
        """Test revolve through session interface."""
        s = ModelingSession()
        points = [(5, 0), (10, 0), (10, 20), (5, 20)]
        data = s.create_revolve(points, 360.0)
        assert data["solids"] == 1
        assert data["faces"] > 2

    def test_loft_session(self):
        """Test loft through session interface."""
        s = ModelingSession()
        sec1 = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
        sec2 = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
        data = s.create_loft([sec1, sec2], [0, 20])
        assert data["solids"] == 1

    def test_shell_session(self):
        """Test shell through session interface."""
        s = ModelingSession()
        s.create_box(60, 40, 30)
        data = s.add_shell(2.0, ">Z")
        assert data["faces"] > 6

    def test_export_stl_session(self, tmp_path):
        """Test STL export through session."""
        s = ModelingSession()
        s.create_box(50, 30, 10)
        out = tmp_path / "session.stl"
        s.export_stl(out)
        assert out.exists()

    def test_housing_workflow(self, tmp_path):
        """Real-world workflow: box → shell → mounting holes → export STL."""
        s = ModelingSession()
        s.create_box(80, 60, 40)
        s.add_shell(2.0, ">Z")
        # Add mounting holes through the bottom
        s.add_hole(25, 20, 4, face_selector="<Z")
        s.add_hole(-25, 20, 4, face_selector="<Z")
        s.add_hole(25, -20, 4, face_selector="<Z")
        s.add_hole(-25, -20, 4, face_selector="<Z")
        g = s.graph
        assert len(g.features) >= 4  # at least the 4 holes
        # Export
        stl_out = str(tmp_path / "housing.stl")
        s.export_stl(stl_out)
        assert Path(stl_out).exists()


class TestToolExecutorAdvanced:
    def test_revolve_tool(self):
        ex = ToolExecutor()
        r = ex.call("create_revolve", {
            "points": [[5, 0], [10, 0], [10, 20], [5, 20]],
            "angle_degrees": 360,
        })
        assert r.success
        assert r.data["solids"] == 1

    def test_loft_tool(self):
        ex = ToolExecutor()
        r = ex.call("create_loft", {
            "sections": [
                [[-10, -10], [10, -10], [10, 10], [-10, 10]],
                [[-5, -5], [5, -5], [5, 5], [-5, 5]],
            ],
            "heights": [0, 20],
        })
        assert r.success
        assert r.data["solids"] == 1

    def test_shell_tool(self):
        ex = ToolExecutor()
        ex.call("create_box", {"length": 60, "width": 40, "height": 30})
        r = ex.call("add_shell", {"thickness": 2.0, "face_selector": ">Z"})
        assert r.success
        assert r.data["faces"] > 6

    def test_export_stl_tool(self, tmp_path):
        ex = ToolExecutor()
        ex.call("create_box", {"length": 50, "width": 30, "height": 10})
        out = str(tmp_path / "tool.stl")
        r = ex.call("export_stl", {"output_path": out})
        assert r.success
        assert Path(out).exists()

    def test_full_housing_workflow(self, tmp_path):
        """AI workflow: create enclosure → shell → drill → export STL + STEP."""
        ex = ToolExecutor()
        ex.call("create_box", {"length": 80, "width": 60, "height": 40})
        ex.call("add_shell", {"thickness": 2.0, "face_selector": ">Z"})
        ex.call("add_hole", {"center_x": 25, "center_y": 20, "diameter": 4, "face_selector": "<Z"})
        ex.call("add_hole", {"center_x": -25, "center_y": 20, "diameter": 4, "face_selector": "<Z"})

        r = ex.call("get_summary", {})
        assert r.data["faces"] > 10

        step_out = str(tmp_path / "housing.step")
        stl_out = str(tmp_path / "housing.stl")
        ex.call("export_step", {"output_path": step_out})
        ex.call("export_stl", {"output_path": stl_out})
        assert Path(step_out).exists()
        assert Path(stl_out).exists()

    def test_tool_count_increased(self):
        """Verify we now have more tools after Phase 4+5+6+sketch+GDT."""
        assert len(TOOL_SCHEMAS) >= 53  # 45 prior + 8 sketch tools


# ---------------------------------------------------------------------------
# Phase 5 — Multi-Body, Assembly, Standard Parts
# ---------------------------------------------------------------------------

class TestDesignRules:
    def test_cnc_check_passes(self):
        """A simple box passes CNC milling rules."""
        s = ModelingSession()
        s.create_box(100, 60, 20)
        result = s.check_design_rules("cnc_milling")
        assert result["passed"] is True

    def test_cnc_check_small_hole(self):
        """A very small hole should trigger a violation."""
        s = ModelingSession()
        s.create_box(100, 60, 20)
        s.add_hole(0, 0, 0.5)  # 0.5mm diameter < 1.0mm min
        result = s.check_design_rules("cnc_milling")
        # Should have at least one violation about min hole diameter
        hole_violations = [v for v in result["violations"] if v["rule"] == "min_hole_diameter"]
        assert len(hole_violations) >= 1

    def test_injection_molding_draft_warning(self):
        """A box without draft should warn for injection molding."""
        s = ModelingSession()
        s.create_box(50, 50, 30)
        result = s.check_design_rules("injection_molding")
        draft_violations = [v for v in result["violations"] if v["rule"] == "min_draft_angle"]
        assert len(draft_violations) >= 1

    def test_fdm_check(self):
        """FDM check on a simple box (no overhangs)."""
        s = ModelingSession()
        s.create_box(50, 50, 20)
        result = s.check_design_rules("fdm_3d_print")
        assert result["passed"] is True

    def test_available_processes(self):
        from next3d.core.design_rules import list_available_processes
        processes = list_available_processes()
        assert "cnc_milling" in processes
        assert "injection_molding" in processes
        assert "fdm_3d_print" in processes
        assert len(processes) >= 6

    def test_check_via_executor(self):
        ex = ToolExecutor()
        ex.call("create_box", {"length": 100, "width": 60, "height": 20})
        r = ex.call("check_design_rules", {"process": "cnc_milling"})
        assert r.success
        assert "passed" in r.data


class TestParametricDimensions:
    def test_set_and_get_parameter(self):
        s = ModelingSession()
        s.set_parameter("wall_thickness", 3.0, "Enclosure wall thickness")
        s.set_parameter("bolt_size", 6.0, "M6 bolt diameter")
        params = s.get_parameters()
        assert params["count"] == 2
        assert params["parameters"]["wall_thickness"]["value"] == 3.0
        assert params["parameters"]["bolt_size"]["value"] == 6.0

    def test_get_single_parameter(self):
        s = ModelingSession()
        s.set_parameter("height", 25.0)
        assert s.get_parameter("height") == 25.0

    def test_parameter_not_found(self):
        s = ModelingSession()
        with pytest.raises(ModelingError):
            s.get_parameter("nonexistent")

    def test_update_parameter(self):
        s = ModelingSession()
        s.set_parameter("wall", 2.0)
        s.set_parameter("wall", 3.0)  # update
        assert s.get_parameter("wall") == 3.0

    def test_parametric_via_executor(self):
        ex = ToolExecutor()
        r = ex.call("set_parameter", {
            "name": "wall_thickness", "value": 2.5,
            "description": "Enclosure wall",
        })
        assert r.success
        r = ex.call("get_parameters", {})
        assert r.data["count"] == 1
        assert r.data["parameters"]["wall_thickness"]["value"] == 2.5


class TestFEA:
    def test_plain_plate_fea(self):
        """FEA on a plain plate without stiffeners."""
        s = ModelingSession()
        result = s.run_fea(
            plate_width=300, plate_height=300, plate_thickness=3.0,
            material="steel_mild", pressure_mpa=0.01, bc_type="fixed_edges",
        )
        assert "max_deflection_mm" in result
        assert "max_plate_stress_MPa" in result
        assert "safety_factor_plate" in result
        assert result["max_deflection_mm"] >= 0

    def test_stiffened_plate_fea(self):
        """FEA on a plate with RHS stiffener grid."""
        s = ModelingSession()
        result = s.run_fea(
            plate_width=304.8, plate_height=304.8, plate_thickness=3.0,
            grid_spacing_x=152.4, grid_spacing_y=152.4,
            rhs_size="50x50x3", material="steel_mild",
            pressure_mpa=0.1, bc_type="fixed_edges",
        )
        assert result["max_deflection_mm"] >= 0
        assert result["mesh"]["beam_elements"] > 0
        assert "verdict" in result

    def test_fea_different_materials(self):
        """FEA with aluminum vs steel."""
        s = ModelingSession()
        steel = s.run_fea(
            plate_width=200, plate_height=200, plate_thickness=5.0,
            material="steel_mild", pressure_mpa=0.05, bc_type="simply_supported",
        )
        alum = s.run_fea(
            plate_width=200, plate_height=200, plate_thickness=5.0,
            material="aluminum_6061", pressure_mpa=0.05, bc_type="simply_supported",
        )
        # Aluminum deflects more (lower E)
        assert alum["max_deflection_mm"] > steel["max_deflection_mm"]

    def test_fea_point_load(self):
        """FEA with a point load at center."""
        s = ModelingSession()
        result = s.run_fea(
            plate_width=200, plate_height=200, plate_thickness=5.0,
            material="steel_mild",
            point_loads=[{"x": 100, "y": 100, "force": 1000}],
            bc_type="simply_supported",
        )
        assert result["center_deflection_mm"] > 0

    def test_fea_via_executor(self):
        """FEA through the tool executor."""
        ex = ToolExecutor()
        r = ex.call("run_fea", {
            "plate_width": 304.8, "plate_height": 304.8,
            "plate_thickness": 3.0,
            "grid_spacing_x": 152.4, "grid_spacing_y": 152.4,
            "rhs_size": "50x50x3",
            "material": "steel_mild",
            "pressure_mpa": 0.1,
            "bc_type": "fixed_edges",
        })
        assert r.success
        assert "max_deflection_mm" in r.data

    def test_fea_parametric(self):
        """Parametric FEA comparison."""
        ex = ToolExecutor()
        r = ex.call("run_fea_parametric", {
            "base_config": {
                "plate_width": 300, "plate_height": 300,
                "plate_thickness": 3, "pressure_mpa": 0.05,
                "rhs_size": "25x25x2",
                "grid_spacing_x": 150, "grid_spacing_y": 150,
            },
            "variations": [
                {"rhs_size": "25x25x2", "label": "Small RHS"},
                {"rhs_size": "50x50x3", "label": "Large RHS"},
            ],
        })
        assert r.success
        assert r.data["config_count"] == 2

    def test_chamber_wall_convenience(self):
        """Test the chamber wall convenience function directly."""
        from next3d.core.fea import analyze_chamber_wall
        result = analyze_chamber_wall(
            plate_width=304.8, plate_height=304.8,
            plate_thickness=3.0, grid_spacing=152.4,
            rhs_size="50x50x3", pressure_mpa=0.1,
        )
        assert "max_deflection_mm" in result
        assert "passes_2mm_limit" in result
        assert "rhs_properties" in result


class TestMultiBody:
    def test_create_named_bodies(self):
        """Create multiple named bodies."""
        s = ModelingSession()
        s.create_body("bracket", "box", length=100, width=60, height=20)
        s.create_body("shaft", "cylinder", radius=5, height=80)
        assert len(s.body_names) == 2
        assert "bracket" in s.body_names
        assert "shaft" in s.body_names

    def test_switch_active_body(self):
        """Switch between bodies and verify isolation."""
        s = ModelingSession()
        s.create_body("a", "box", length=50, width=50, height=10)
        s.create_body("b", "box", length=100, width=80, height=30)
        s.set_active_body("a")
        g_a = s.graph
        s.set_active_body("b")
        g_b = s.graph
        # Both are boxes with 6 faces, but different sizes
        assert len(g_a.faces) == 6
        assert len(g_b.faces) == 6

    def test_modify_specific_body(self):
        """Modify only the active body, other body unaffected."""
        s = ModelingSession()
        s.create_body("plate", "box", length=100, width=60, height=10)
        s.create_body("block", "box", length=50, width=50, height=50)
        s.set_active_body("plate")
        s.add_hole(0, 0, 10)
        # Plate has a hole, block doesn't
        g_plate = s.graph
        assert len(g_plate.faces) > 6
        s.set_active_body("block")
        g_block = s.graph
        assert len(g_block.faces) == 6

    def test_list_bodies(self):
        s = ModelingSession()
        s.create_body("part_a", "box", material="aluminum", length=50, width=30, height=10)
        s.create_body("part_b", "cylinder", material="steel", radius=10, height=40)
        data = s.list_bodies()
        assert data["count"] == 2
        names = [b["name"] for b in data["bodies"]]
        assert "part_a" in names
        assert "part_b" in names
        # Check material assignment
        mats = {b["name"]: b["material"] for b in data["bodies"]}
        assert mats["part_a"] == "aluminum"
        assert mats["part_b"] == "steel"

    def test_delete_body(self):
        s = ModelingSession()
        s.create_body("temp", "box", length=10, width=10, height=10)
        s.create_body("keep", "box", length=20, width=20, height=20)
        s.delete_body("temp")
        assert "temp" not in s.body_names
        assert "keep" in s.body_names

    def test_duplicate_body(self):
        s = ModelingSession()
        s.create_body("original", "box", length=50, width=30, height=10)
        s.duplicate_body("original", "copy")
        assert len(s.body_names) == 2
        # Both should have same face count
        s.set_active_body("original")
        f1 = len(s.graph.faces)
        s.set_active_body("copy")
        f2 = len(s.graph.faces)
        assert f1 == f2

    def test_backward_compat_single_body(self):
        """Legacy single-body workflow still works."""
        s = ModelingSession()
        s.create_box(100, 60, 20)
        s.add_hole(0, 0, 10)
        g = s.graph
        assert len(g.faces) > 6
        assert s.active_body_name == "default"


class TestAssembly:
    def test_place_body(self):
        s = ModelingSession()
        s.create_body("base", "box", length=100, width=100, height=10)
        s.create_body("pillar", "cylinder", radius=5, height=50)
        data = s.place_body("pillar", x=20, y=20, z=10)
        assert data["body"] == "pillar"
        assert data["placement"]["translation"]["z"] == 10

    def test_export_assembly(self, tmp_path):
        s = ModelingSession()
        s.create_body("base", "box", length=100, width=100, height=10)
        s.create_body("pillar", "cylinder", radius=5, height=50)
        s.place_body("pillar", z=10)
        out = tmp_path / "assembly.step"
        s.export_assembly(out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_check_interference_clear(self):
        """Two separated bodies should not interfere."""
        s = ModelingSession()
        s.create_body("a", "box", length=10, width=10, height=10)
        s.create_body("b", "box", length=10, width=10, height=10)
        s.place_body("b", x=50)  # far apart
        result = s.check_interference("a", "b")
        assert result["interferes"] is False

    def test_check_interference_overlap(self):
        """Two overlapping bodies should interfere."""
        s = ModelingSession()
        s.create_body("a", "box", length=20, width=20, height=20)
        s.create_body("b", "box", length=20, width=20, height=20)
        # Both at origin = full overlap
        result = s.check_interference("a", "b")
        assert result["interferes"] is True
        assert result["interference_volume_mm3"] > 0

    def test_bom(self):
        s = ModelingSession()
        s.create_body("plate", "box", material="aluminum", length=100, width=60, height=5)
        s.create_body("bracket", "box", material="steel", length=30, width=20, height=40)
        bom = s.get_bom()
        assert bom["item_count"] == 2
        assert bom["total_mass_grams"] > 0
        names = [item["name"] for item in bom["items"]]
        assert "plate" in names
        assert "bracket" in names

    def test_add_mate_constraint(self):
        s = ModelingSession()
        s.create_body("a", "box", length=50, width=50, height=10)
        s.create_body("b", "box", length=50, width=50, height=10)
        data = s.add_mate("coincident", "a", "face_top", "b", "face_bottom")
        assert data["mate_type"] == "coincident"
        assert data["total_mates"] == 1


class TestStandardParts:
    def test_hex_bolt(self):
        from next3d.parts.fasteners import iso_hex_bolt
        shape = iso_hex_bolt("M6", 30)
        g = build_semantic_graph_from_shape(shape)
        assert len(g.solids) == 1
        assert len(g.faces) > 4

    def test_hex_nut(self):
        from next3d.parts.fasteners import iso_hex_nut
        shape = iso_hex_nut("M8")
        g = build_semantic_graph_from_shape(shape)
        assert len(g.solids) == 1

    def test_flat_washer(self):
        from next3d.parts.fasteners import iso_flat_washer
        shape = iso_flat_washer("M6")
        g = build_semantic_graph_from_shape(shape)
        assert len(g.solids) == 1

    def test_socket_head_cap_screw(self):
        from next3d.parts.fasteners import iso_socket_head_cap_screw
        shape = iso_socket_head_cap_screw("M5", 25)
        g = build_semantic_graph_from_shape(shape)
        assert len(g.solids) == 1

    def test_add_standard_part_tool(self):
        ex = ToolExecutor()
        r = ex.call("add_standard_part", {
            "name": "bolt_1", "part_type": "hex_bolt", "size": "M6", "length": 25,
        })
        assert r.success
        assert r.data["name"] == "bolt_1"

    def test_list_sizes(self):
        from next3d.parts.fasteners import list_available_sizes
        sizes = list_available_sizes()
        assert "M6" in sizes
        assert "M8" in sizes
        assert len(sizes) >= 7


class TestMultiBodyToolExecutor:
    def test_create_and_list(self):
        ex = ToolExecutor()
        ex.call("create_named_body", {
            "name": "plate", "shape_type": "box",
            "length": 100, "width": 60, "height": 10,
        })
        ex.call("create_named_body", {
            "name": "shaft", "shape_type": "cylinder",
            "radius": 5, "height": 50,
        })
        r = ex.call("list_bodies", {})
        assert r.success
        assert r.data["count"] == 2

    def test_switch_and_modify(self):
        ex = ToolExecutor()
        ex.call("create_named_body", {
            "name": "plate", "shape_type": "box",
            "length": 100, "width": 60, "height": 10,
        })
        ex.call("create_named_body", {
            "name": "block", "shape_type": "box",
            "length": 50, "width": 50, "height": 50,
        })
        ex.call("set_active_body", {"name": "plate"})
        r = ex.call("add_hole", {"center_x": 0, "center_y": 0, "diameter": 10})
        assert r.success
        assert r.data["body"] == "plate"
        assert r.data["faces"] > 6

    def test_place_and_export(self, tmp_path):
        ex = ToolExecutor()
        ex.call("create_named_body", {
            "name": "base", "shape_type": "box",
            "length": 100, "width": 100, "height": 10,
        })
        ex.call("create_named_body", {
            "name": "column", "shape_type": "cylinder",
            "radius": 8, "height": 60,
        })
        ex.call("place_body", {"name": "column", "z": 10})
        out = str(tmp_path / "assembly.step")
        r = ex.call("export_assembly", {"output_path": out})
        assert r.success
        assert Path(out).exists()

    def test_interference_check(self):
        ex = ToolExecutor()
        ex.call("create_named_body", {
            "name": "a", "shape_type": "box",
            "length": 20, "width": 20, "height": 20,
        })
        ex.call("create_named_body", {
            "name": "b", "shape_type": "box",
            "length": 20, "width": 20, "height": 20,
        })
        r = ex.call("check_interference", {"body_a": "a", "body_b": "b"})
        assert r.success
        assert r.data["interferes"] is True

    def test_bom_tool(self):
        ex = ToolExecutor()
        ex.call("create_named_body", {
            "name": "plate", "shape_type": "box", "material": "aluminum",
            "length": 100, "width": 60, "height": 5,
        })
        r = ex.call("get_bom", {})
        assert r.success
        assert r.data["item_count"] == 1
        assert r.data["items"][0]["material"] == "aluminum"

    def test_full_assembly_workflow(self, tmp_path):
        """Complete assembly workflow: parts → place → check → BOM → export."""
        ex = ToolExecutor()

        # Create parts
        ex.call("create_named_body", {
            "name": "base_plate", "shape_type": "box", "material": "aluminum",
            "length": 100, "width": 80, "height": 5,
        })
        ex.call("set_active_body", {"name": "base_plate"})
        ex.call("add_hole", {"center_x": 30, "center_y": 25, "diameter": 6.4})
        ex.call("add_hole", {"center_x": -30, "center_y": 25, "diameter": 6.4})

        # Add standard fasteners
        ex.call("add_standard_part", {
            "name": "bolt_1", "part_type": "hex_bolt", "size": "M6", "length": 20,
        })
        ex.call("add_standard_part", {
            "name": "bolt_2", "part_type": "hex_bolt", "size": "M6", "length": 20,
        })

        # Place bolts
        ex.call("place_body", {"name": "bolt_1", "x": 30, "y": 25, "z": 5})
        ex.call("place_body", {"name": "bolt_2", "x": -30, "y": 25, "z": 5})

        # Check BOM
        r = ex.call("get_bom", {})
        assert r.data["item_count"] == 3  # plate + 2 bolts

        # Export assembly
        out = str(tmp_path / "full_assembly.step")
        r = ex.call("export_assembly", {"output_path": out})
        assert r.success
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# Session tests
# ---------------------------------------------------------------------------

class TestSession:
    def test_create_and_query(self):
        s = ModelingSession()
        s.create_box(100, 60, 20)
        g = s.graph
        assert len(g.faces) == 6
        assert s.history.length == 1

    def test_modify(self):
        s = ModelingSession()
        s.create_box(100, 60, 20)
        s.add_hole(0, 0, 10)
        g = s.graph
        holes = [f for f in g.features if "hole" in f.feature_type.value]
        assert len(holes) >= 1
        assert s.history.length == 2

    def test_undo(self):
        s = ModelingSession()
        s.create_box(100, 60, 20)
        s.add_hole(0, 0, 10)
        assert len(s.graph.faces) > 6
        s.undo()
        assert len(s.graph.faces) == 6
        assert s.history.length == 1

    def test_undo_empty(self):
        s = ModelingSession()
        with pytest.raises(ModelingError):
            s.undo()

    def test_modify_empty(self):
        s = ModelingSession()
        with pytest.raises(ModelingError):
            s.add_hole(0, 0, 10)

    def test_export_step(self, tmp_path):
        s = ModelingSession()
        s.create_box(100, 60, 20)
        s.add_hole(0, 0, 10)
        out = tmp_path / "test.step"
        s.export_step(out)
        assert out.exists()
        assert out.stat().st_size > 0
        # Round-trip: re-read and verify
        g = build_semantic_graph_from_shape(
            __import__("next3d.core.brep", fromlist=["load_step"]).load_step(out).shape
        )
        holes = [f for f in g.features if "hole" in f.feature_type.value]
        assert len(holes) >= 1

    def test_load_step(self):
        s = ModelingSession()
        s.load_step(FIXTURES / "sample_block.step")
        g = s.graph
        assert len(g.faces) > 0
        assert s.history.length == 1

    def test_to_script(self):
        s = ModelingSession()
        s.create_box(100, 60, 20)
        s.add_hole(0, 0, 10)
        script = s.to_script()
        assert "box" in script
        assert "hole" in script
        assert "import cadquery" in script

    def test_summary(self):
        s = ModelingSession()
        s.create_box(100, 60, 20)
        summary = s.summary()
        assert summary["operations"] == 1
        assert summary["faces"] == 6

    def test_multi_step_workflow(self):
        """Full AI workflow: create → fillet → drill → pocket → export."""
        s = ModelingSession()
        s.create_box(100, 60, 20)
        s.add_fillet(1.0, "|Z")  # fillet first, before subtractive ops
        s.add_hole(20, 10, 8)
        s.add_hole(-20, -10, 8)
        s.add_pocket(0, 0, 40, 20, 5)
        g = s.graph
        assert len(g.features) > 0
        assert s.history.length == 5


# ---------------------------------------------------------------------------
# Operation log tests
# ---------------------------------------------------------------------------

class TestOperationLog:
    def test_append_and_pop(self):
        log = OperationLog()
        op = Operation(op_type=OpType.CREATE_BOX, params={"length": 100, "width": 60, "height": 20})
        log.append(op)
        assert log.length == 1
        popped = log.pop()
        assert popped.op_type == OpType.CREATE_BOX
        assert log.length == 0

    def test_cadquery_script(self):
        log = OperationLog()
        log.append(Operation(op_type=OpType.CREATE_BOX, params={"length": 100, "width": 60, "height": 20}))
        log.append(Operation(op_type=OpType.ADD_HOLE, params={"center_x": 0, "center_y": 0, "diameter": 10}))
        script = log.to_cadquery_script()
        assert "box(100, 60, 20)" in script
        assert "hole(10)" in script

    def test_serializable(self):
        log = OperationLog()
        log.append(Operation(op_type=OpType.CREATE_BOX, params={"length": 100, "width": 60, "height": 20}))
        j = log.model_dump_json()
        log2 = OperationLog.model_validate_json(j)
        assert log2.length == 1


# ---------------------------------------------------------------------------
# Tool executor tests
# ---------------------------------------------------------------------------

class TestToolExecutor:
    def test_create_and_query(self):
        ex = ToolExecutor()
        r = ex.call("create_box", {"length": 100, "width": 60, "height": 20})
        assert r.success
        assert r.data["faces"] == 6

        r2 = ex.call("get_summary", {})
        assert r2.success
        assert r2.data["faces"] == 6

    def test_modify_and_query(self):
        ex = ToolExecutor()
        ex.call("create_box", {"length": 100, "width": 60, "height": 20})
        r = ex.call("add_hole", {"center_x": 0, "center_y": 0, "diameter": 10})
        assert r.success
        assert r.data["faces"] > 6

        r2 = ex.call("get_features", {"feature_type": "through_hole"})
        assert r2.success
        assert r2.data["count"] >= 1

    def test_find_faces(self):
        ex = ToolExecutor()
        ex.call("create_box", {"length": 100, "width": 60, "height": 20})
        r = ex.call("find_faces", {"surface_type": "plane", "normal_z": 1.0})
        assert r.success
        assert r.data["count"] >= 1  # top face

    def test_undo(self):
        ex = ToolExecutor()
        ex.call("create_box", {"length": 100, "width": 60, "height": 20})
        ex.call("add_hole", {"center_x": 0, "center_y": 0, "diameter": 10})
        r = ex.call("undo", {})
        assert r.success
        r2 = ex.call("get_summary", {})
        assert r2.data["faces"] == 6  # back to box

    def test_unknown_tool(self):
        ex = ToolExecutor()
        r = ex.call("nonexistent_tool", {})
        assert not r.success

    def test_invalid_params(self):
        ex = ToolExecutor()
        r = ex.call("create_box", {"length": -5, "width": 60, "height": 20})
        assert not r.success

    def test_export_script(self):
        ex = ToolExecutor()
        ex.call("create_box", {"length": 100, "width": 60, "height": 20})
        r = ex.call("export_script", {})
        assert r.success
        assert "box" in r.data["script"]

    def test_export_step(self, tmp_path):
        ex = ToolExecutor()
        ex.call("create_box", {"length": 100, "width": 60, "height": 20})
        out = str(tmp_path / "test.step")
        r = ex.call("export_step", {"output_path": out})
        assert r.success
        assert Path(out).exists()

    def test_load_step(self):
        ex = ToolExecutor()
        r = ex.call("load_step", {"path": str(FIXTURES / "sample_block.step")})
        assert r.success
        r2 = ex.call("get_summary", {})
        assert r2.data["faces"] > 0

    def test_full_ai_workflow(self, tmp_path):
        """Simulate what an AI agent would do: create → modify → query → export."""
        ex = ToolExecutor()

        # Step 1: Create base shape
        ex.call("create_box", {"length": 80, "width": 50, "height": 15})

        # Step 2: Query to understand
        r = ex.call("get_summary", {})
        assert r.data["faces"] == 6

        # Step 3: Find the top face
        r = ex.call("find_faces", {"surface_type": "plane", "normal_z": 1.0})
        assert r.data["count"] >= 1

        # Step 4: Add features
        ex.call("add_hole", {"center_x": 20, "center_y": 10, "diameter": 6})
        ex.call("add_hole", {"center_x": -20, "center_y": 10, "diameter": 6})
        ex.call("add_hole", {"center_x": 20, "center_y": -10, "diameter": 6})
        ex.call("add_hole", {"center_x": -20, "center_y": -10, "diameter": 6})

        # Step 5: Verify features
        r = ex.call("get_features", {"feature_type": "through_hole"})
        assert r.data["count"] == 4

        # Step 6: Add chamfers
        ex.call("add_chamfer", {"distance": 1.0, "edge_selector": "|Z"})

        # Step 7: Export
        out = str(tmp_path / "ai_part.step")
        r = ex.call("export_step", {"output_path": out})
        assert r.success

        # Step 8: Get CadQuery script
        r = ex.call("export_script", {})
        assert "hole" in r.data["script"]


# ---------------------------------------------------------------------------
# Sketch tests
# ---------------------------------------------------------------------------

class TestSketch:
    def test_sketch_line_and_extrude(self):
        """Create sketch with lines forming a closed triangle, extrude."""
        s = ModelingSession()
        s.create_sketch("XY")
        s.sketch_add_line(0, 0, 10, 0)
        s.sketch_add_line(10, 0, 5, 10)
        s.sketch_add_line(5, 10, 0, 0)
        data = s.sketch_extrude(5)
        assert data["solids"] == 1
        assert data["faces"] >= 5  # 2 triangular + 3 side faces

    def test_sketch_circle_extrude(self):
        """Sketch a circle, extrude to make a cylinder."""
        s = ModelingSession()
        s.create_sketch("XY")
        s.sketch_add_circle(0, 0, 10)
        data = s.sketch_extrude(20)
        assert data["solids"] == 1
        assert data["faces"] >= 3  # top, bottom, curved

    def test_sketch_rect_extrude(self):
        """Sketch a rectangle, extrude to make a box."""
        s = ModelingSession()
        s.create_sketch("XY")
        s.sketch_add_rect(0, 0, 30, 20)
        data = s.sketch_extrude(10)
        assert data["solids"] == 1
        assert data["faces"] == 6  # box

    def test_sketch_with_constraints(self):
        """Add constraints and verify they are stored."""
        s = ModelingSession()
        s.create_sketch("XY")
        l1 = s.sketch_add_line(0, 0, 10, 0)["entity_id"]
        l2 = s.sketch_add_line(10, 0, 10, 10)["entity_id"]
        c1 = s.sketch_add_constraint("horizontal", l1)
        assert "constraint_id" in c1
        c2 = s.sketch_add_constraint("vertical", l2)
        assert c2["constraint_count"] == 2
        c3 = s.sketch_add_constraint("perpendicular", l1, l2)
        assert c3["constraint_count"] == 3
        c4 = s.sketch_add_constraint("distance", l1, value=10.0)
        assert c4["constraint_count"] == 4

    def test_sketch_revolve(self):
        """Sketch an L-shaped profile, revolve to make a solid of revolution."""
        s = ModelingSession()
        s.create_sketch("XY")
        # Profile: a rectangle offset from the Y axis
        s.sketch_add_line(5, 0, 10, 0)
        s.sketch_add_line(10, 0, 10, 20)
        s.sketch_add_line(10, 20, 5, 20)
        s.sketch_add_line(5, 20, 5, 0)
        data = s.sketch_revolve(360.0, axis_origin=(0, 0), axis_direction=(0, 1))
        assert data["solids"] == 1
        assert data["faces"] >= 2

    def test_sketch_via_executor(self):
        """Test sketch workflow through ToolExecutor."""
        ex = ToolExecutor()
        r = ex.call("create_sketch", {"plane": "XY"})
        assert r.success
        assert r.data["status"] == "sketch_active"

        r = ex.call("sketch_add_circle", {"cx": 0, "cy": 0, "radius": 15})
        assert r.success
        assert r.data["entity_count"] == 1

        r = ex.call("sketch_extrude", {"height": 25})
        assert r.success
        assert r.data["solids"] == 1
        assert r.data["faces"] >= 3


# ---------------------------------------------------------------------------
# GD&T tests
# ---------------------------------------------------------------------------

class TestGDT:
    def test_add_datum_and_tolerance(self):
        """Add datums and tolerances, verify they are stored."""
        s = ModelingSession()
        s.create_box(100, 60, 20)
        g = s.graph
        # Pick the first planar face as datum
        plane_faces = [f for f in g.faces if f.surface_type.value == "plane"]
        assert len(plane_faces) >= 1
        face_id = plane_faces[0].persistent_id

        # Add datum
        data = s.add_datum("A", face_id, "Primary datum")
        assert data["label"] == "A"
        assert data["total_datums"] == 1

        # Add tolerance
        data = s.add_tolerance("flatness", 0.05, face_id, description="Surface flatness")
        assert data["tolerance_type"] == "flatness"
        assert data["total_tolerances"] == 1

    def test_get_gdt(self):
        """Get GD&T annotations and verify structure."""
        s = ModelingSession()
        s.create_box(100, 60, 20)
        g = s.graph
        face_id = g.faces[0].persistent_id

        s.add_datum("A", face_id, "Primary datum")
        s.add_tolerance("flatness", 0.05, face_id)
        s.add_tolerance("position", 0.1, face_id, datum_refs=["A"], material_condition="MMC")

        gdt = s.get_gdt()
        assert gdt["datum_count"] == 1
        assert gdt["tolerance_count"] == 2
        assert gdt["body"] == "default"
        assert len(gdt["datums"]) == 1
        assert gdt["datums"][0]["label"] == "A"
        assert len(gdt["tolerances"]) == 2
        # No validation issues since entity_id is valid and datum is defined
        assert len(gdt["validation_issues"]) == 0

    def test_suggest_gdt(self):
        """Create a box with holes, verify auto-suggestions."""
        s = ModelingSession()
        s.create_box(100, 60, 20)
        s.add_hole(20, 10, 8)
        s.add_hole(-20, -10, 8)

        suggestions = s.suggest_gdt()
        assert suggestions["datum_count"] >= 1  # at least datum A
        assert suggestions["tolerance_count"] >= 1  # at least flatness or position
        # Should suggest datum A for the largest planar face
        assert any(d["label"] == "A" for d in suggestions["datums"])
        # Should suggest position for through holes
        position_tols = [
            t for t in suggestions["tolerances"]
            if t["tolerance_type"] == "position"
        ]
        assert len(position_tols) >= 1

    def test_gdt_via_executor(self):
        """Test GD&T through the ToolExecutor."""
        ex = ToolExecutor()
        ex.call("create_box", {"length": 100, "width": 60, "height": 20})

        # Get a face ID from features/faces
        r = ex.call("find_faces", {"surface_type": "plane", "normal_z": 1.0})
        assert r.success
        face_id = r.data["faces"][0]["id"]

        # Add datum
        r = ex.call("add_datum", {"label": "A", "entity_id": face_id, "description": "Top face"})
        assert r.success
        assert r.data["label"] == "A"

        # Add tolerance
        r = ex.call("add_tolerance", {
            "tolerance_type": "flatness", "value": 0.05,
            "entity_id": face_id,
        })
        assert r.success
        assert r.data["tolerance_type"] == "flatness"

        # Get GD&T
        r = ex.call("get_gdt", {})
        assert r.success
        assert r.data["datum_count"] == 1
        assert r.data["tolerance_count"] == 1

        # Suggest GD&T
        r = ex.call("suggest_gdt", {})
        assert r.success
        assert r.data["datum_count"] >= 1

    def test_validate_gdt_missing_datum(self):
        """Tolerance referencing undefined datum should produce validation error."""
        s = ModelingSession()
        s.create_box(100, 60, 20)
        g = s.graph
        face_id = g.faces[0].persistent_id

        # Add tolerance referencing datum "B" without defining it
        s.add_tolerance("position", 0.1, face_id, datum_refs=["B"])

        gdt = s.get_gdt()
        issues = gdt["validation_issues"]
        # Should have an error about datum "B" not being defined
        datum_errors = [
            i for i in issues
            if "datum" in i["message"].lower() and "B" in i["message"]
        ]
        assert len(datum_errors) >= 1
        assert datum_errors[0]["severity"] == "error"


# ---------------------------------------------------------------------------
# Tool schema export tests
# ---------------------------------------------------------------------------

class TestToolSchemas:
    def test_openai_format(self):
        tools = to_openai_tools()
        assert len(tools) == len(TOOL_SCHEMAS)
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_mcp_format(self):
        tools = to_mcp_tools()
        assert len(tools) == len(TOOL_SCHEMAS)
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    def test_anthropic_format(self):
        tools = to_anthropic_tools()
        assert len(tools) == len(TOOL_SCHEMAS)
        for tool in tools:
            assert "name" in tool
            assert "input_schema" in tool

    def test_schemas_are_valid_json(self):
        for fmt in ["openai", "mcp", "anthropic"]:
            from next3d.tools.formats import to_json
            j = to_json(fmt)
            parsed = json.loads(j)
            assert isinstance(parsed, list)

    def test_all_tools_have_descriptions(self):
        for name, cls in TOOL_SCHEMAS.items():
            assert cls.__doc__, f"Tool {name} missing docstring"


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestModelingCLI:
    def test_tools_openai(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["tools", "--format", "openai"])
        assert result.exit_code == 0
        tools = json.loads(result.output)
        assert len(tools) > 10

    def test_tools_mcp(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["tools", "--format", "mcp"])
        assert result.exit_code == 0
        tools = json.loads(result.output)
        assert len(tools) > 10

    def test_call_create_box(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "call", "create_box", '{"length":100,"width":60,"height":20}'
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["success"]

    def test_build_pipeline(self, tmp_path):
        runner = CliRunner()
        out = str(tmp_path / "built.step")
        pipeline = json.dumps([
            {"tool": "create_box", "params": {"length": 100, "width": 60, "height": 20}},
            {"tool": "add_hole", "params": {"center_x": 0, "center_y": 0, "diameter": 10}},
        ])
        result = runner.invoke(cli, ["build", pipeline, "-o", out])
        assert result.exit_code == 0
        assert Path(out).exists()

    def test_build_with_script_export(self, tmp_path):
        runner = CliRunner()
        out = str(tmp_path / "built.step")
        script_out = str(tmp_path / "script.py")
        pipeline = json.dumps([
            {"tool": "create_box", "params": {"length": 50, "width": 50, "height": 10}},
        ])
        result = runner.invoke(cli, ["build", pipeline, "-o", out, "--export-script", script_out])
        assert result.exit_code == 0
        script = Path(script_out).read_text()
        assert "box" in script


# ---------------------------------------------------------------------------
# Topology Optimization tests
# ---------------------------------------------------------------------------

class TestTopologyOptimization:
    def test_add_loads_and_bcs(self):
        """Add loads and boundary conditions, verify they are stored."""
        s = ModelingSession()
        s.create_box(100, 60, 20)

        # Add a load
        data = s.add_load("downward", 0, 0, -1000, 0, 0, 10)
        assert data["name"] == "downward"
        assert data["total_loads"] == 1
        assert data["force"]["z"] == -1000

        # Add a second load
        data = s.add_load("lateral", 500, 0, 0, 50, 0, 10)
        assert data["total_loads"] == 2

        # Add boundary conditions
        data = s.add_boundary_condition("fixed_base", "fixed", "<Z")
        assert data["name"] == "fixed_base"
        assert data["bc_type"] == "fixed"
        assert data["total_boundary_conditions"] == 1

        data = s.add_boundary_condition("roller_side", "roller", "<X")
        assert data["total_boundary_conditions"] == 2

    def test_run_optimization(self):
        """Create box, add load on top, fix bottom, run optimization."""
        s = ModelingSession()
        s.create_box(100, 60, 20)

        # Load on top center
        s.add_load("top_load", 0, 0, -500, 0, 0, 10)
        # Fix bottom face
        s.add_boundary_condition("fixed_bottom", "fixed", "<Z")

        result = s.run_topology_optimization(volume_fraction=0.3, resolution=5)

        assert "original_volume" in result
        assert "optimized_volume" in result
        assert "volume_reduction_pct" in result
        assert "removal_regions" in result
        assert "density_field" in result
        assert "suggestions" in result
        assert result["body"] == "default"

    def test_optimization_reduces_volume(self):
        """Verify optimization achieves volume reduction."""
        s = ModelingSession()
        s.create_box(100, 60, 20)

        # Load on top center
        s.add_load("top_load", 0, 0, -500, 0, 0, 10)
        # Fix bottom face
        s.add_boundary_condition("fixed_bottom", "fixed", "<Z")

        result = s.run_topology_optimization(volume_fraction=0.3, resolution=6)

        # Should achieve some volume reduction
        assert result["volume_reduction_pct"] > 0
        assert result["optimized_volume"] < result["original_volume"]
        assert result["removal_regions_count"] > 0

    def test_optimization_via_executor(self):
        """Test the full topology optimization workflow through ToolExecutor."""
        ex = ToolExecutor()

        # Create geometry
        r = ex.call("create_box", {"length": 100, "width": 60, "height": 20})
        assert r.success

        # Add load
        r = ex.call("add_load", {
            "name": "top_load", "fx": 0, "fy": 0, "fz": -500,
            "px": 0, "py": 0, "pz": 10,
        })
        assert r.success
        assert r.data["total_loads"] == 1

        # Add boundary condition
        r = ex.call("add_boundary_condition", {
            "name": "fixed_base", "bc_type": "fixed", "face_selector": "<Z",
        })
        assert r.success
        assert r.data["total_boundary_conditions"] == 1

        # Run optimization
        r = ex.call("run_topology_optimization", {
            "volume_fraction": 0.4, "resolution": 5,
        })
        assert r.success
        assert "volume_reduction_pct" in r.data
        assert r.data["total_voxels"] > 0

    def test_optimization_result_structure(self):
        """Verify result has all expected fields with correct types."""
        s = ModelingSession()
        s.create_box(80, 50, 30)

        s.add_load("force_1", 0, 0, -1000, 0, 0, 15)
        s.add_boundary_condition("support", "fixed", "<Z")

        result = s.run_topology_optimization(volume_fraction=0.5, resolution=5)

        # Top-level fields
        assert isinstance(result["original_volume"], float)
        assert isinstance(result["optimized_volume"], float)
        assert isinstance(result["volume_reduction_pct"], float)
        assert isinstance(result["removal_regions_count"], int)
        assert isinstance(result["total_voxels"], int)
        assert isinstance(result["kept_voxels"], int)
        assert isinstance(result["removal_regions"], list)
        assert isinstance(result["density_field"], list)
        assert isinstance(result["suggestions"], list)
        assert isinstance(result["body"], str)

        # Density field voxel structure
        if result["density_field"]:
            voxel = result["density_field"][0]
            assert "x" in voxel
            assert "y" in voxel
            assert "z" in voxel
            assert "density" in voxel
            assert "stress_proxy" in voxel

        # Removal region structure
        if result["removal_regions"]:
            region = result["removal_regions"][0]
            assert "x" in region
            assert "y" in region
            assert "z" in region
            assert "size" in region
            assert "reason" in region

        # Suggestions are non-empty strings
        assert len(result["suggestions"]) > 0
        for s_text in result["suggestions"]:
            assert isinstance(s_text, str)
            assert len(s_text) > 0

        # Volume consistency
        assert result["total_voxels"] == result["kept_voxels"] + result["removal_regions_count"]
