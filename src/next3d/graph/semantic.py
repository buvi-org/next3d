"""Semantic graph assembly — the 3D DOM.

Combines topology graph, feature recognition results, and relationships
into a single queryable SemanticGraph.
"""

from __future__ import annotations

from pathlib import Path

from next3d.core.brep import load_step
from next3d.core.identity import solid_id
from next3d.core.schema import SemanticGraph, SolidData, Vec3
from next3d.core.topology import build_topology_graph
import next3d.features  # noqa: F401 — triggers recognizer registration
from next3d.features.engine import recognize_all

from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS


def build_semantic_graph(step_path: str | Path) -> SemanticGraph:
    """Build the complete semantic graph from a STEP file.

    This is the main entry point for the system. It:
    1. Loads the STEP file
    2. Builds the topology graph
    3. Runs feature recognition
    4. Assembles everything into a SemanticGraph

    Args:
        step_path: Path to a STEP file.

    Returns:
        A SemanticGraph — the '3D DOM' of the part.
    """
    model = load_step(step_path)

    # Build topology
    _nx_graph, faces, edges, vertices, adjacency = build_topology_graph(model.shape)

    # Extract solids
    solids = []
    sexp = TopExp_Explorer(model.shape, TopAbs_SOLID)
    while sexp.More():
        s = sexp.Current()
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(s, props)
        volume = props.Mass()
        cp = props.CentreOfMass()
        centroid = Vec3(x=cp.X(), y=cp.Y(), z=cp.Z())
        pid = solid_id(centroid.x, centroid.y, centroid.z, volume)

        # Collect face IDs belonging to this solid
        face_ids = [f.persistent_id for f in faces]  # simplified: assign all faces to first solid

        solids.append(
            SolidData(
                persistent_id=pid,
                face_ids=face_ids,
                volume=volume,
                centroid=centroid,
            )
        )
        sexp.Next()

    # Recognize features
    features = recognize_all(faces, edges, adjacency)

    return SemanticGraph(
        solids=solids,
        faces=faces,
        edges=edges,
        vertices=vertices,
        features=features,
        adjacency=adjacency,
        relationships=[],  # Phase 2
    )
