"""Semantic graph assembly — the 3D DOM.

Combines topology graph, feature recognition results, and relationships
into a single queryable SemanticGraph.
"""

from __future__ import annotations

from pathlib import Path

from next3d.core.brep import load_step
from next3d.core.identity import solid_id
from next3d.core.patterns import detect_linear_patterns, detect_symmetric_features
from next3d.core.relationships import detect_all_relationships
from next3d.core.schema import SemanticGraph, SolidData, Vec3
from next3d.core.topology import build_topology_graph
import next3d.features  # noqa: F401 — triggers recognizer registration
from next3d.features.engine import recognize_all

from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
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

    # Extract solids with proper face-to-solid mapping
    solids = []
    face_hash_to_pid = {}
    for f in faces:
        # We'll match faces by centroid+area fingerprint
        key = (round(f.centroid.x, 4), round(f.centroid.y, 4), round(f.centroid.z, 4), round(f.area, 4))
        face_hash_to_pid[key] = f.persistent_id

    sexp = TopExp_Explorer(model.shape, TopAbs_SOLID)
    while sexp.More():
        s = sexp.Current()
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(s, props)
        volume = props.Mass()
        cp = props.CentreOfMass()
        centroid = Vec3(x=cp.X(), y=cp.Y(), z=cp.Z())
        pid = solid_id(centroid.x, centroid.y, centroid.z, volume)

        # Collect face IDs belonging to THIS solid by iterating its faces
        solid_face_ids = []
        fexp = TopExp_Explorer(s, TopAbs_FACE)
        while fexp.More():
            face_shape = fexp.Current()
            fprops = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face_shape, fprops)
            farea = fprops.Mass()
            fcp = fprops.CentreOfMass()
            key = (round(fcp.X(), 4), round(fcp.Y(), 4), round(fcp.Z(), 4), round(farea, 4))
            fpid = face_hash_to_pid.get(key)
            if fpid and fpid not in solid_face_ids:
                solid_face_ids.append(fpid)
            fexp.Next()

        solids.append(
            SolidData(
                persistent_id=pid,
                face_ids=solid_face_ids,
                volume=volume,
                centroid=centroid,
            )
        )
        sexp.Next()

    # Recognize features
    features = recognize_all(faces, edges, adjacency)

    # Detect geometric relationships
    relationships = detect_all_relationships(faces, adjacency)

    # Detect inter-feature relationships (symmetry, patterns)
    face_lookup = {f.persistent_id: f for f in faces}
    relationships.extend(detect_symmetric_features(features, face_lookup))
    relationships.extend(detect_linear_patterns(features, face_lookup))

    return SemanticGraph(
        solids=solids,
        faces=faces,
        edges=edges,
        vertices=vertices,
        features=features,
        adjacency=adjacency,
        relationships=relationships,
    )
