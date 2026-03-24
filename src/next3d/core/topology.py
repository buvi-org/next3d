"""Topology graph builder: B-Rep shape → NetworkX graph.

Traverses the OpenCascade B-Rep structure and constructs a graph where
nodes are topological entities (Solid, Face, Edge, Vertex) and edges
represent containment, adjacency, and shared-vertex relationships.
"""

from __future__ import annotations

import networkx as nx

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import (
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    GeomAbs_BSplineSurface,
    GeomAbs_Line,
    GeomAbs_Circle,
    GeomAbs_Ellipse,
    GeomAbs_BSplineCurve,
)
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer, TopExp
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCP.TopoDS import TopoDS, TopoDS_Shape

from next3d.core.identity import vertex_id, edge_id, face_id
from next3d.core.schema import (
    SurfaceType,
    CurveType,
    Vec3,
    VertexData,
    EdgeData,
    FaceData,
    AdjacencyEdge,
    EdgeRelationType,
)


# ---------------------------------------------------------------------------
# Surface / curve type mapping
# ---------------------------------------------------------------------------

_SURFACE_TYPE_MAP = {
    GeomAbs_Plane: SurfaceType.PLANE,
    GeomAbs_Cylinder: SurfaceType.CYLINDER,
    GeomAbs_Cone: SurfaceType.CONE,
    GeomAbs_Sphere: SurfaceType.SPHERE,
    GeomAbs_Torus: SurfaceType.TORUS,
    GeomAbs_BSplineSurface: SurfaceType.BSPLINE,
}

_CURVE_TYPE_MAP = {
    GeomAbs_Line: CurveType.LINE,
    GeomAbs_Circle: CurveType.CIRCLE,
    GeomAbs_Ellipse: CurveType.ELLIPSE,
    GeomAbs_BSplineCurve: CurveType.BSPLINE,
}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _classify_surface(face: TopoDS_Shape) -> tuple[SurfaceType, dict]:
    """Classify a face's surface type and extract geometric parameters."""
    adaptor = BRepAdaptor_Surface(TopoDS.Face_s(face))
    stype = _SURFACE_TYPE_MAP.get(adaptor.GetType(), SurfaceType.OTHER)
    params: dict = {}

    if stype == SurfaceType.PLANE:
        pln = adaptor.Plane()
        ax = pln.Axis()
        d = ax.Direction()
        params["normal"] = Vec3(x=d.X(), y=d.Y(), z=d.Z())
    elif stype == SurfaceType.CYLINDER:
        cyl = adaptor.Cylinder()
        params["radius"] = cyl.Radius()
        ax = cyl.Axis()
        d = ax.Direction()
        params["axis"] = Vec3(x=d.X(), y=d.Y(), z=d.Z())
    elif stype == SurfaceType.CONE:
        cone = adaptor.Cone()
        params["radius"] = cone.RefRadius()
        ax = cone.Axis()
        d = ax.Direction()
        params["axis"] = Vec3(x=d.X(), y=d.Y(), z=d.Z())
    elif stype == SurfaceType.SPHERE:
        sph = adaptor.Sphere()
        params["radius"] = sph.Radius()
    elif stype == SurfaceType.TORUS:
        tor = adaptor.Torus()
        params["radius"] = tor.MajorRadius()
        ax = tor.Axis()
        d = ax.Direction()
        params["axis"] = Vec3(x=d.X(), y=d.Y(), z=d.Z())

    return stype, params


def _classify_curve(edge: TopoDS_Shape) -> tuple[CurveType, dict]:
    """Classify an edge's curve type and extract geometric parameters."""
    adaptor = BRepAdaptor_Curve(TopoDS.Edge_s(edge))
    ctype = _CURVE_TYPE_MAP.get(adaptor.GetType(), CurveType.OTHER)
    params: dict = {}

    if ctype == CurveType.CIRCLE:
        circ = adaptor.Circle()
        params["radius"] = circ.Radius()
        center = circ.Location()
        params["center"] = Vec3(x=center.X(), y=center.Y(), z=center.Z())

    return ctype, params


def _face_centroid_and_area(face: TopoDS_Shape) -> tuple[Vec3, float]:
    """Compute the centroid and area of a face."""
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    area = props.Mass()
    cp = props.CentreOfMass()
    return Vec3(x=cp.X(), y=cp.Y(), z=cp.Z()), area


def _edge_length(edge: TopoDS_Shape) -> float:
    """Compute the length of an edge."""
    props = GProp_GProps()
    BRepGProp.LinearProperties_s(edge, props)
    return props.Mass()


def _vertex_point(vertex: TopoDS_Shape) -> Vec3:
    """Extract the 3D point from a vertex."""
    pt = BRep_Tool.Pnt_s(TopoDS.Vertex_s(vertex))
    return Vec3(x=pt.X(), y=pt.Y(), z=pt.Z())


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_topology_graph(
    shape: TopoDS_Shape,
) -> tuple[nx.DiGraph, list[FaceData], list[EdgeData], list[VertexData], list[AdjacencyEdge]]:
    """Build a topology graph from an OpenCascade shape.

    Returns:
        - NetworkX directed graph (nodes=entities, edges=containment/adjacency)
        - Lists of FaceData, EdgeData, VertexData (with persistent IDs)
        - List of AdjacencyEdge records
    """
    graph = nx.DiGraph()

    # Deduplicate by OpenCascade shape hash
    seen_vertices: dict[int, VertexData] = {}
    seen_edges: dict[int, EdgeData] = {}
    seen_faces: dict[int, FaceData] = {}

    # --- Vertices ---
    vexp = TopExp_Explorer(shape, TopAbs_VERTEX)
    while vexp.More():
        v = vexp.Current()
        key = v.HashCode(2**31 - 1)
        if key not in seen_vertices:
            pt = _vertex_point(v)
            pid = vertex_id(pt.x, pt.y, pt.z)
            vdata = VertexData(persistent_id=pid, position=pt)
            seen_vertices[key] = vdata
            graph.add_node(pid, entity_type="vertex")
        vexp.Next()

    # --- Edges ---
    eexp = TopExp_Explorer(shape, TopAbs_EDGE)
    while eexp.More():
        e = eexp.Current()
        key = e.HashCode(2**31 - 1)
        if key not in seen_edges:
            ctype, cparams = _classify_curve(e)
            length = _edge_length(e)

            # Get start/end vertices
            v_first = TopExp.FirstVertex_s(TopoDS.Edge_s(e))
            v_last = TopExp.LastVertex_s(TopoDS.Edge_s(e))
            pt_first = _vertex_point(v_first)
            pt_last = _vertex_point(v_last)
            sv_id = vertex_id(pt_first.x, pt_first.y, pt_first.z)
            ev_id = vertex_id(pt_last.x, pt_last.y, pt_last.z)

            pid = edge_id(
                pt_first.x, pt_first.y, pt_first.z,
                pt_last.x, pt_last.y, pt_last.z,
                ctype.value,
            )
            edata = EdgeData(
                persistent_id=pid,
                curve_type=ctype,
                length=length,
                start_vertex=sv_id,
                end_vertex=ev_id,
                radius=cparams.get("radius"),
                center=cparams.get("center"),
            )
            seen_edges[key] = edata
            graph.add_node(pid, entity_type="edge")
            graph.add_edge(pid, sv_id, relation="contains")
            graph.add_edge(pid, ev_id, relation="contains")
        eexp.Next()

    # --- Faces ---
    fexp = TopExp_Explorer(shape, TopAbs_FACE)
    while fexp.More():
        f = fexp.Current()
        key = f.HashCode(2**31 - 1)
        if key not in seen_faces:
            stype, sparams = _classify_surface(f)
            centroid, area = _face_centroid_and_area(f)
            pid = face_id(stype.value, centroid.x, centroid.y, centroid.z, area)

            # Collect bounding edge IDs
            bound_edge_ids = []
            edge_exp = TopExp_Explorer(f, TopAbs_EDGE)
            while edge_exp.More():
                ekey = edge_exp.Current().HashCode(2**31 - 1)
                if ekey in seen_edges:
                    bound_edge_ids.append(seen_edges[ekey].persistent_id)
                edge_exp.Next()

            fdata = FaceData(
                persistent_id=pid,
                surface_type=stype,
                area=area,
                centroid=centroid,
                normal=sparams.get("normal"),
                radius=sparams.get("radius"),
                axis=sparams.get("axis"),
                edge_ids=bound_edge_ids,
            )
            seen_faces[key] = fdata
            graph.add_node(pid, entity_type="face")
            for eid in bound_edge_ids:
                graph.add_edge(pid, eid, relation="contains")
        fexp.Next()

    # --- Face adjacency (faces sharing an edge) ---
    adjacency_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, adjacency_map)

    adjacency_edges: list[AdjacencyEdge] = []
    for i in range(1, adjacency_map.Extent() + 1):
        edge_shape = adjacency_map.FindKey(i)
        face_list = adjacency_map.FindFromIndex(i)

        ekey = edge_shape.HashCode(2**31 - 1)
        shared_eid = seen_edges[ekey].persistent_id if ekey in seen_edges else None

        face_ids = []
        it = face_list.begin()
        while it != face_list.end():
            fkey = it.HashCode(2**31 - 1)
            if fkey in seen_faces:
                face_ids.append(seen_faces[fkey].persistent_id)
            it = it.Next() if hasattr(it, "Next") else None
            if it is None:
                break

        # Create adjacency edges for each pair of faces sharing this edge
        for j in range(len(face_ids)):
            for k in range(j + 1, len(face_ids)):
                adj = AdjacencyEdge(
                    source_id=face_ids[j],
                    target_id=face_ids[k],
                    edge_type=EdgeRelationType.ADJACENT,
                    shared_edge_id=shared_eid,
                )
                adjacency_edges.append(adj)
                graph.add_edge(face_ids[j], face_ids[k], relation="adjacent")
                graph.add_edge(face_ids[k], face_ids[j], relation="adjacent")

    return (
        graph,
        list(seen_faces.values()),
        list(seen_edges.values()),
        list(seen_vertices.values()),
        adjacency_edges,
    )
