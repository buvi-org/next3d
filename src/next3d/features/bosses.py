"""Boss/protrusion feature recognition.

Detection strategy:
- Cylindrical face with a planar top cap
- The cylindrical face is also adjacent to a larger planar base face
- Boss axis is along the cylinder axis
"""

from __future__ import annotations

from next3d.core.identity import feature_id
from next3d.core.schema import (
    AdjacencyEdge,
    EdgeData,
    EdgeRelationType,
    FaceData,
    FeatureData,
    FeatureType,
    SurfaceType,
)


class BossRecognizer:
    """Recognizes boss features (cylindrical protrusions from a base face).

    Pattern: cylindrical face adjacent to both a small planar top cap
    and a larger planar base. Distinguished from holes by the top cap
    being smaller than the base.
    """

    def recognize(
        self,
        faces: list[FaceData],
        edges: list[EdgeData],
        adjacency: list[AdjacencyEdge],
    ) -> list[FeatureData]:
        features: list[FeatureData] = []
        face_lookup = {f.persistent_id: f for f in faces}

        adj_lookup: dict[str, list[str]] = {}
        for adj in adjacency:
            if adj.edge_type != EdgeRelationType.ADJACENT:
                continue
            adj_lookup.setdefault(adj.source_id, []).append(adj.target_id)
            adj_lookup.setdefault(adj.target_id, []).append(adj.source_id)

        used_faces: set[str] = set()

        for face in faces:
            if face.persistent_id in used_faces:
                continue
            if face.surface_type != SurfaceType.CYLINDER:
                continue
            if face.radius is None:
                continue

            adjacent_ids = adj_lookup.get(face.persistent_id, [])
            adjacent_faces = [face_lookup[fid] for fid in adjacent_ids if fid in face_lookup]

            # Find adjacent planar faces
            planar_neighbors = [
                af for af in adjacent_faces if af.surface_type == SurfaceType.PLANE
            ]

            if len(planar_neighbors) < 2:
                continue

            # Sort by area: smallest is the top cap, largest is the base
            planar_neighbors.sort(key=lambda f: f.area)
            cap = planar_neighbors[0]
            base = planar_neighbors[-1]

            # Boss criteria:
            # 1. Cap area should be roughly pi*r^2 (circular top)
            # 2. Base should be significantly larger than the cap
            # 3. Cylinder area suggests a protrusion (not a hole)
            import math
            expected_cap_area = math.pi * face.radius ** 2
            cap_area_ratio = cap.area / expected_cap_area if expected_cap_area > 0 else 0

            if base.area <= cap.area * 2:
                continue  # base not significantly larger
            if cap_area_ratio < 0.5 or cap_area_ratio > 2.0:
                continue  # cap doesn't match circular top

            boss_faces = [face.persistent_id, cap.persistent_id]
            if any(fid in used_faces for fid in boss_faces):
                continue

            # Estimate height from cylinder area: A = 2*pi*r*h → h = A/(2*pi*r)
            height = face.area / (2 * math.pi * face.radius) if face.radius > 0 else 0

            fid = feature_id(FeatureType.BOSS.value, boss_faces)
            features.append(
                FeatureData(
                    persistent_id=fid,
                    feature_type=FeatureType.BOSS,
                    face_ids=boss_faces + [base.persistent_id],
                    edge_ids=face.edge_ids,
                    parameters={
                        "diameter": face.radius * 2,
                        "radius": face.radius,
                        "height": round(height, 3),
                    },
                    axis=face.axis,
                    description=f"Boss, diameter {face.radius * 2:.2f}mm, height ~{height:.1f}mm",
                )
            )
            used_faces.update(boss_faces)

        return features
