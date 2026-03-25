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
        """Verify we now have more tools after Phase 4."""
        assert len(TOOL_SCHEMAS) >= 32  # was 22, now 32 with Phase 4 additions


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
