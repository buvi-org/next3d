"""Inter-feature relationship detection: symmetry and patterns.

Detects:
- Symmetric features (mirrored across a plane)
- Linear/circular patterns (repeated features at regular spacing)
"""

from __future__ import annotations

import math

from next3d.core.schema import (
    FeatureData,
    Relationship,
    RelationshipType,
    Vec3,
)


def _centroid_of_feature(feat: FeatureData, face_lookup: dict) -> Vec3 | None:
    """Approximate feature centroid from constituent face centroids."""
    if not feat.face_ids:
        return None
    xs, ys, zs = [], [], []
    for fid in feat.face_ids:
        face = face_lookup.get(fid)
        if face:
            xs.append(face.centroid.x)
            ys.append(face.centroid.y)
            zs.append(face.centroid.z)
    if not xs:
        return None
    return Vec3(x=sum(xs) / len(xs), y=sum(ys) / len(ys), z=sum(zs) / len(zs))


def _distance(a: Vec3, b: Vec3) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _reflect_point(p: Vec3, plane_normal: Vec3, plane_point: Vec3) -> Vec3:
    """Reflect a point across a plane defined by normal + point."""
    # d = dot(p - plane_point, normal)
    dx = p.x - plane_point.x
    dy = p.y - plane_point.y
    dz = p.z - plane_point.z
    d = dx * plane_normal.x + dy * plane_normal.y + dz * plane_normal.z
    return Vec3(
        x=p.x - 2 * d * plane_normal.x,
        y=p.y - 2 * d * plane_normal.y,
        z=p.z - 2 * d * plane_normal.z,
    )


def detect_symmetric_features(
    features: list[FeatureData],
    face_lookup: dict,
    tolerance: float = 0.5,
) -> list[Relationship]:
    """Detect features that are symmetric about principal planes (XY, XZ, YZ).

    Two features are symmetric if:
    1. They have the same type
    2. They have the same parameters (within tolerance)
    3. One's centroid is the mirror of the other across a principal plane
    """
    results: list[Relationship] = []
    seen: set[tuple[str, str]] = set()

    # Principal plane normals and origins
    planes = [
        (Vec3(x=1, y=0, z=0), Vec3(x=0, y=0, z=0), "YZ"),
        (Vec3(x=0, y=1, z=0), Vec3(x=0, y=0, z=0), "XZ"),
        (Vec3(x=0, y=0, z=1), Vec3(x=0, y=0, z=0), "XY"),
    ]

    # Compute feature centroids
    centroids: dict[str, Vec3] = {}
    for feat in features:
        c = _centroid_of_feature(feat, face_lookup)
        if c:
            centroids[feat.persistent_id] = c

    for i, f1 in enumerate(features):
        c1 = centroids.get(f1.persistent_id)
        if c1 is None:
            continue

        for j in range(i + 1, len(features)):
            f2 = features[j]
            if f1.feature_type != f2.feature_type:
                continue

            c2 = centroids.get(f2.persistent_id)
            if c2 is None:
                continue

            pair = tuple(sorted([f1.persistent_id, f2.persistent_id]))
            if pair in seen:
                continue

            # Check parameter similarity
            params_match = True
            for key in f1.parameters:
                v1 = f1.parameters.get(key, 0)
                v2 = f2.parameters.get(key, 0)
                if abs(v1 - v2) > tolerance:
                    params_match = False
                    break
            if not params_match:
                continue

            # Check symmetry across each principal plane
            for normal, origin, plane_name in planes:
                reflected = _reflect_point(c1, normal, origin)
                dist = _distance(reflected, c2)
                if dist < tolerance:
                    results.append(
                        Relationship(
                            source_id=f1.persistent_id,
                            target_id=f2.persistent_id,
                            relationship_type=RelationshipType.SYMMETRIC,
                            parameters={"plane": plane_name},
                        )
                    )
                    seen.add(pair)
                    break

    return results


def detect_linear_patterns(
    features: list[FeatureData],
    face_lookup: dict,
    tolerance: float = 0.5,
) -> list[Relationship]:
    """Detect linear patterns of identical features.

    Groups features of the same type+params that are evenly spaced along a line.
    """
    results: list[Relationship] = []

    # Group features by type and parameter signature
    groups: dict[str, list[FeatureData]] = {}
    for feat in features:
        key = feat.feature_type.value + "|" + str(sorted(feat.parameters.items()))
        groups.setdefault(key, []).append(feat)

    centroids: dict[str, Vec3] = {}
    for feat in features:
        c = _centroid_of_feature(feat, face_lookup)
        if c:
            centroids[feat.persistent_id] = c

    for key, group in groups.items():
        if len(group) < 3:
            continue  # need at least 3 for a pattern

        # Sort by distance from first feature
        c0 = centroids.get(group[0].persistent_id)
        if c0 is None:
            continue

        with_dist = []
        for feat in group:
            c = centroids.get(feat.persistent_id)
            if c:
                with_dist.append((feat, _distance(c0, c)))
        with_dist.sort(key=lambda x: x[1])

        # Check for even spacing
        if len(with_dist) < 3:
            continue

        distances = [with_dist[i + 1][1] - with_dist[i][1] for i in range(len(with_dist) - 1)]
        avg_spacing = sum(distances) / len(distances)

        if avg_spacing < 0.1:
            continue

        all_even = all(abs(d - avg_spacing) < tolerance for d in distances)
        if all_even:
            # Mark all as pattern members
            for i in range(len(with_dist) - 1):
                results.append(
                    Relationship(
                        source_id=with_dist[i][0].persistent_id,
                        target_id=with_dist[i + 1][0].persistent_id,
                        relationship_type=RelationshipType.PATTERN_MEMBER,
                        parameters={"spacing": round(avg_spacing, 3)},
                    )
                )

    return results
