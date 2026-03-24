"""Tests for the MCP server — verifies tool listing and execution over protocol."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

FIXTURES = Path(__file__).parent / "fixtures"

# Server params reused across tests
SERVER = StdioServerParameters(
    command="python",
    args=["-m", "next3d.mcp"],
    env={"PYTHONPATH": "src"},
)


@pytest.fixture()
def run_async():
    """Helper to run async test functions."""
    def _run(coro):
        return asyncio.run(coro)
    return _run


class TestMCPToolListing:
    def test_lists_all_tools(self, run_async):
        async def _test():
            async with stdio_client(SERVER) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    names = {t.name for t in tools.tools}
                    assert len(names) >= 22
                    # Verify key tools exist
                    assert "create_box" in names
                    assert "add_hole" in names
                    assert "get_summary" in names
                    assert "export_step" in names
                    assert "undo" in names
        run_async(_test())

    def test_tools_have_descriptions(self, run_async):
        async def _test():
            async with stdio_client(SERVER) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    for tool in tools.tools:
                        assert tool.description, f"{tool.name} missing description"
                        assert tool.inputSchema, f"{tool.name} missing input schema"
        run_async(_test())


class TestMCPToolExecution:
    def test_create_and_query(self, run_async):
        async def _test():
            async with stdio_client(SERVER) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Create
                    r = await session.call_tool("create_box", {
                        "length": 100, "width": 60, "height": 20,
                    })
                    text = r.content[0].text
                    assert "Created box" in text
                    assert '"faces": 6' in text

                    # Query
                    r = await session.call_tool("get_summary", {})
                    text = r.content[0].text
                    assert '"faces": 6' in text
        run_async(_test())

    def test_modify_and_query_features(self, run_async):
        async def _test():
            async with stdio_client(SERVER) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    await session.call_tool("create_box", {
                        "length": 100, "width": 60, "height": 20,
                    })
                    await session.call_tool("add_hole", {
                        "center_x": 0, "center_y": 0, "diameter": 10,
                    })

                    r = await session.call_tool("get_features", {
                        "feature_type": "through_hole",
                    })
                    text = r.content[0].text
                    assert '"count": 1' in text
                    assert "through_hole" in text
        run_async(_test())

    def test_find_faces(self, run_async):
        async def _test():
            async with stdio_client(SERVER) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    await session.call_tool("create_box", {
                        "length": 100, "width": 60, "height": 20,
                    })

                    r = await session.call_tool("find_faces", {
                        "surface_type": "plane", "normal_z": 1.0,
                    })
                    text = r.content[0].text
                    assert '"count":' in text
                    assert "plane" in text
        run_async(_test())

    def test_undo(self, run_async):
        async def _test():
            async with stdio_client(SERVER) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    await session.call_tool("create_box", {
                        "length": 100, "width": 60, "height": 20,
                    })
                    await session.call_tool("add_hole", {
                        "center_x": 0, "center_y": 0, "diameter": 10,
                    })
                    r = await session.call_tool("undo", {})
                    text = r.content[0].text
                    assert "Undone" in text
                    assert '"faces": 6' in text
        run_async(_test())

    def test_export_script(self, run_async):
        async def _test():
            async with stdio_client(SERVER) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    await session.call_tool("create_box", {
                        "length": 100, "width": 60, "height": 20,
                    })
                    r = await session.call_tool("export_script", {})
                    text = r.content[0].text
                    assert "import cadquery" in text
                    assert "box" in text
        run_async(_test())

    def test_export_step(self, run_async, tmp_path):
        async def _test():
            async with stdio_client(SERVER) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    await session.call_tool("create_box", {
                        "length": 100, "width": 60, "height": 20,
                    })
                    out = str(tmp_path / "mcp_test.step")
                    r = await session.call_tool("export_step", {"output_path": out})
                    text = r.content[0].text
                    assert "Exported" in text
                    assert Path(out).exists()
        run_async(_test())

    def test_full_ai_workflow(self, run_async, tmp_path):
        """Simulate a complete AI agent session over MCP."""
        async def _test():
            async with stdio_client(SERVER) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # AI creates base shape
                    await session.call_tool("create_box", {
                        "length": 80, "width": 50, "height": 15,
                    })

                    # AI queries to understand
                    r = await session.call_tool("get_summary", {})
                    assert '"faces": 6' in r.content[0].text

                    # AI finds top face
                    r = await session.call_tool("find_faces", {
                        "surface_type": "plane", "normal_z": 1.0,
                    })

                    # AI adds mounting holes
                    await session.call_tool("add_hole", {"center_x": 25, "center_y": 15, "diameter": 5})
                    await session.call_tool("add_hole", {"center_x": -25, "center_y": 15, "diameter": 5})
                    await session.call_tool("add_hole", {"center_x": 25, "center_y": -15, "diameter": 5})
                    await session.call_tool("add_hole", {"center_x": -25, "center_y": -15, "diameter": 5})

                    # AI verifies holes
                    r = await session.call_tool("get_features", {"feature_type": "through_hole"})
                    assert '"count": 4' in r.content[0].text

                    # AI adds pocket
                    await session.call_tool("add_pocket", {
                        "center_x": 0, "center_y": 0, "length": 30, "width": 20, "depth": 5,
                    })

                    # AI exports
                    out = str(tmp_path / "ai_bracket.step")
                    await session.call_tool("export_step", {"output_path": out})
                    assert Path(out).exists()

                    # AI gets the CadQuery script
                    r = await session.call_tool("export_script", {})
                    assert "hole" in r.content[0].text
                    assert "box" in r.content[0].text
        run_async(_test())

    def test_load_step_file(self, run_async):
        async def _test():
            async with stdio_client(SERVER) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    r = await session.call_tool("load_step", {
                        "path": str(FIXTURES / "sample_block.step"),
                    })
                    text = r.content[0].text
                    assert "Loaded" in text

                    r = await session.call_tool("get_summary", {})
                    # Should have faces from the loaded file
                    assert '"faces":' in r.content[0].text
        run_async(_test())
