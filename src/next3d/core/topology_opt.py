"""Simplified topology optimization — distance-based SIMP-like approach.

This is NOT a real FEA solver. It uses geometric heuristics to approximate
load paths and identify low-stress regions for material removal. The goal
is to give AI agents a meaningful optimization workflow without requiring
scipy or a full finite-element library.

Algorithm:
1. Voxelize the solid's bounding box at a given resolution
2. Classify each voxel as inside/outside the solid
3. For each interior voxel, compute a "stress proxy" based on:
   - Proximity to load application points (higher stress near loads)
   - Proximity to fixed boundary conditions (higher stress near supports)
   - Position along the load path (connecting loads to constraints)
4. Assign density values: high-stress voxels → density 1.0, low-stress → 0.0
5. Threshold by volume fraction to select removal regions
6. Return density field, removal regions, and human-readable suggestions
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from OCP.TopoDS import TopoDS_Shape

from next3d.core.spatial import BoundingBox, bounding_box, point_in_solid
from next3d.core.schema import Vec3
from next3d.modeling import kernel


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LoadCase:
    """A force applied at a point on the geometry."""

    name: str
    force: tuple[float, float, float]  # Fx, Fy, Fz in Newtons
    application_point: tuple[float, float, float]  # where force is applied


@dataclass
class BoundaryCondition:
    """A support condition on a face of the geometry."""

    name: str
    bc_type: str  # "fixed", "pinned", "roller"
    face_selector: str  # CadQuery face selector like ">Z", "<Z"


@dataclass
class OptimizationSetup:
    """Configuration for a topology optimization run."""

    loads: list[LoadCase]
    constraints: list[BoundaryCondition]
    volume_fraction: float  # target: 0.3 = keep 30% of material
    material_youngs_modulus: float = 200000.0  # MPa (steel default)
    material_poisson_ratio: float = 0.3


@dataclass
class OptimizationResult:
    """Results from a topology optimization run."""

    original_volume: float
    optimized_volume: float
    volume_reduction_pct: float
    removal_regions: list[dict[str, Any]]  # regions suggested for removal
    density_field: list[dict[str, Any]]  # voxel positions + density values
    suggestions: list[str]  # human-readable suggestions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_optimization(
    loads: list[LoadCase],
    constraints: list[BoundaryCondition],
    volume_fraction: float = 0.3,
    youngs_modulus: float = 200000.0,
    poisson_ratio: float = 0.3,
) -> OptimizationSetup:
    """Create an optimization setup from loads and boundary conditions.

    Args:
        loads: List of LoadCase objects defining applied forces.
        constraints: List of BoundaryCondition objects defining supports.
        volume_fraction: Target volume fraction (0.3 = keep 30%).
        youngs_modulus: Material Young's modulus in MPa.
        poisson_ratio: Material Poisson's ratio.

    Returns:
        OptimizationSetup ready for run_optimization.
    """
    if not loads:
        raise ValueError("At least one load case is required.")
    if not constraints:
        raise ValueError("At least one boundary condition is required.")
    if not 0.0 < volume_fraction < 1.0:
        raise ValueError("Volume fraction must be between 0 and 1 (exclusive).")

    return OptimizationSetup(
        loads=loads,
        constraints=constraints,
        volume_fraction=volume_fraction,
        material_youngs_modulus=youngs_modulus,
        material_poisson_ratio=poisson_ratio,
    )


def run_optimization(
    shape: TopoDS_Shape,
    setup: OptimizationSetup,
    resolution: int = 10,
) -> OptimizationResult:
    """Run simplified topology optimization on a shape.

    Voxelizes the shape, computes a stress proxy for each interior voxel
    based on distance to loads and boundary conditions, then marks
    low-stress voxels as removable.

    Args:
        shape: The solid to optimize.
        setup: OptimizationSetup with loads, constraints, volume fraction.
        resolution: Number of voxels along the longest axis.

    Returns:
        OptimizationResult with density field, removal regions, suggestions.
    """
    if resolution < 3:
        raise ValueError("Resolution must be at least 3.")

    # Step 1: Compute bounding box and voxel grid
    bbox = bounding_box(shape)
    sx, sy, sz = bbox.size.x, bbox.size.y, bbox.size.z
    max_dim = max(sx, sy, sz)
    if max_dim < 1e-6:
        raise ValueError("Shape has zero size.")

    voxel_size = max_dim / resolution
    nx = max(1, int(math.ceil(sx / voxel_size)))
    ny = max(1, int(math.ceil(sy / voxel_size)))
    nz = max(1, int(math.ceil(sz / voxel_size)))

    # Step 2: Identify interior voxels
    interior_voxels: list[tuple[float, float, float]] = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                cx = bbox.x_min + (ix + 0.5) * voxel_size
                cy = bbox.y_min + (iy + 0.5) * voxel_size
                cz = bbox.z_min + (iz + 0.5) * voxel_size
                pt = Vec3(x=cx, y=cy, z=cz)
                state = point_in_solid(shape, pt, tolerance=voxel_size * 0.1)
                if state in ("inside", "on_boundary"):
                    interior_voxels.append((cx, cy, cz))

    if not interior_voxels:
        # Fallback: shape may be too small for voxelization
        return OptimizationResult(
            original_volume=0,
            optimized_volume=0,
            volume_reduction_pct=0,
            removal_regions=[],
            density_field=[],
            suggestions=["Shape too small or resolution too low for voxelization."],
        )

    # Step 3: Compute stress proxy for each voxel
    # Load points and constraint reference points
    load_points = [
        (lc.application_point[0], lc.application_point[1], lc.application_point[2])
        for lc in setup.loads
    ]
    load_magnitudes = [
        math.sqrt(lc.force[0] ** 2 + lc.force[1] ** 2 + lc.force[2] ** 2)
        for lc in setup.loads
    ]

    # Derive constraint reference points from face selectors
    constraint_points = _constraint_reference_points(bbox, setup.constraints)

    # For each voxel, compute stress proxy
    stress_values: list[float] = []
    for vx, vy, vz in interior_voxels:
        stress = _compute_stress_proxy(
            vx, vy, vz, load_points, load_magnitudes, constraint_points, bbox,
        )
        stress_values.append(stress)

    # Step 4: Normalize stress values to [0, 1]
    max_stress = max(stress_values) if stress_values else 1.0
    min_stress = min(stress_values) if stress_values else 0.0
    stress_range = max_stress - min_stress
    if stress_range < 1e-12:
        # Uniform stress — keep everything
        densities = [1.0] * len(stress_values)
    else:
        densities = [(s - min_stress) / stress_range for s in stress_values]

    # Step 5: Apply volume fraction threshold
    # Sort densities to find cutoff
    sorted_densities = sorted(densities)
    cutoff_idx = int(len(sorted_densities) * (1.0 - setup.volume_fraction))
    cutoff_idx = min(cutoff_idx, len(sorted_densities) - 1)
    density_threshold = sorted_densities[cutoff_idx]

    # Assign final densities: above threshold = keep (1.0), below = remove (0.0)
    final_densities = [1.0 if d >= density_threshold else 0.0 for d in densities]

    # Step 6: Build density field and removal regions
    voxel_vol = voxel_size ** 3
    density_field = []
    removal_regions = []
    kept_count = 0

    for i, (vx, vy, vz) in enumerate(interior_voxels):
        density_field.append({
            "x": round(vx, 3),
            "y": round(vy, 3),
            "z": round(vz, 3),
            "density": round(final_densities[i], 3),
            "stress_proxy": round(densities[i], 3),
        })
        if final_densities[i] < 0.5:
            removal_regions.append({
                "x": round(vx, 3),
                "y": round(vy, 3),
                "z": round(vz, 3),
                "size": round(voxel_size, 3),
                "reason": "low_stress",
            })
        else:
            kept_count += 1

    original_volume = len(interior_voxels) * voxel_vol
    optimized_volume = kept_count * voxel_vol
    volume_reduction_pct = round(
        (1.0 - optimized_volume / original_volume) * 100 if original_volume > 0 else 0,
        1,
    )

    # Step 7: Generate suggestions
    suggestions = _generate_suggestions(
        setup, volume_reduction_pct, len(removal_regions), len(interior_voxels),
    )

    return OptimizationResult(
        original_volume=round(original_volume, 2),
        optimized_volume=round(optimized_volume, 2),
        volume_reduction_pct=volume_reduction_pct,
        removal_regions=removal_regions,
        density_field=density_field,
        suggestions=suggestions,
    )


def apply_optimization(
    shape: TopoDS_Shape,
    result: OptimizationResult,
) -> TopoDS_Shape:
    """Apply optimization by cutting removal regions from the shape.

    Creates box-shaped cuts at each removal region to approximate
    the optimized topology.

    Args:
        shape: Original solid shape.
        result: OptimizationResult from run_optimization.

    Returns:
        Modified shape with low-density regions removed.
    """
    current = shape
    for region in result.removal_regions:
        size = region["size"]
        try:
            cut_box = kernel.create_box(
                size, size, size,
                center=(region["x"], region["y"], region["z"]),
            )
            current = kernel.boolean_cut(current, cut_box)
        except Exception:
            # Some cuts may fail near boundaries — skip them
            continue
    return current


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _constraint_reference_points(
    bbox: BoundingBox,
    constraints: list[BoundaryCondition],
) -> list[tuple[float, float, float]]:
    """Derive reference points from face selectors.

    Maps CadQuery face selectors to the center of the corresponding
    bounding box face.
    """
    cx, cy, cz = bbox.center.x, bbox.center.y, bbox.center.z
    face_map: dict[str, tuple[float, float, float]] = {
        ">X": (bbox.x_max, cy, cz),
        "<X": (bbox.x_min, cy, cz),
        ">Y": (cx, bbox.y_max, cz),
        "<Y": (cx, bbox.y_min, cz),
        ">Z": (cx, cy, bbox.z_max),
        "<Z": (cx, cy, bbox.z_min),
    }

    points = []
    for bc in constraints:
        pt = face_map.get(bc.face_selector)
        if pt is not None:
            points.append(pt)
        else:
            # Default: use center of bottom face for unknown selectors
            points.append((cx, cy, bbox.z_min))
    return points


def _compute_stress_proxy(
    vx: float,
    vy: float,
    vz: float,
    load_points: list[tuple[float, float, float]],
    load_magnitudes: list[float],
    constraint_points: list[tuple[float, float, float]],
    bbox: BoundingBox,
) -> float:
    """Compute a simplified stress proxy for a voxel.

    The heuristic: stress is high in regions that are on the "load path"
    between applied loads and fixed supports. We approximate this as:

    stress ~ sum_over_loads(
        load_magnitude / (dist_to_load + eps)
        * 1 / (dist_to_nearest_constraint + eps)
    )

    Plus a bonus for voxels that are close to the straight-line path
    from any load to any constraint (the direct load path).
    """
    diag = bbox.diagonal
    eps = diag * 0.01  # avoid division by zero

    stress = 0.0

    for li, (lx, ly, lz) in enumerate(load_points):
        mag = load_magnitudes[li] if li < len(load_magnitudes) else 1.0
        dist_to_load = math.sqrt((vx - lx) ** 2 + (vy - ly) ** 2 + (vz - lz) ** 2)

        # Contribution from proximity to load
        load_proximity = mag / (dist_to_load + eps)

        for (cx, cy, cz) in constraint_points:
            dist_to_constraint = math.sqrt(
                (vx - cx) ** 2 + (vy - cy) ** 2 + (vz - cz) ** 2
            )

            # Contribution from proximity to constraint
            constraint_proximity = 1.0 / (dist_to_constraint + eps)

            # Load path bonus: how close is the voxel to the line
            # from load to constraint?
            path_bonus = _point_to_line_proximity(
                vx, vy, vz, lx, ly, lz, cx, cy, cz, diag,
            )

            stress += load_proximity * constraint_proximity * (1.0 + path_bonus)

    return stress


def _point_to_line_proximity(
    px: float, py: float, pz: float,
    lx: float, ly: float, lz: float,
    cx: float, cy: float, cz: float,
    diag: float,
) -> float:
    """Compute a proximity bonus for being near the line from (lx,ly,lz) to (cx,cy,cz).

    Returns a value in [0, 1] where 1 = on the line, 0 = far away.
    """
    # Direction vector of line segment
    dx, dy, dz = cx - lx, cy - ly, cz - lz
    seg_len_sq = dx * dx + dy * dy + dz * dz

    if seg_len_sq < 1e-12:
        # Load and constraint at same point
        return 0.0

    # Project point onto line
    t = ((px - lx) * dx + (py - ly) * dy + (pz - lz) * dz) / seg_len_sq
    t = max(0.0, min(1.0, t))  # clamp to segment

    # Closest point on segment
    closest_x = lx + t * dx
    closest_y = ly + t * dy
    closest_z = lz + t * dz

    dist = math.sqrt(
        (px - closest_x) ** 2 + (py - closest_y) ** 2 + (pz - closest_z) ** 2
    )

    # Convert distance to proximity: closer = higher bonus
    # Use a Gaussian-like falloff with width proportional to diagonal
    sigma = diag * 0.15
    proximity = math.exp(-0.5 * (dist / sigma) ** 2)
    return proximity


def _generate_suggestions(
    setup: OptimizationSetup,
    volume_reduction_pct: float,
    num_removal_regions: int,
    total_voxels: int,
) -> list[str]:
    """Generate human-readable suggestions from optimization results."""
    suggestions = []

    if volume_reduction_pct > 0:
        suggestions.append(
            f"Topology optimization suggests {volume_reduction_pct}% volume reduction "
            f"(target was {round((1.0 - setup.volume_fraction) * 100, 1)}%)."
        )
    else:
        suggestions.append("No volume reduction possible — all regions are load-bearing.")

    if num_removal_regions > 0:
        suggestions.append(
            f"{num_removal_regions} of {total_voxels} voxels identified as removable."
        )

    if len(setup.loads) == 1:
        lc = setup.loads[0]
        mag = math.sqrt(lc.force[0] ** 2 + lc.force[1] ** 2 + lc.force[2] ** 2)
        suggestions.append(
            f"Single load case '{lc.name}': {mag:.0f}N at "
            f"({lc.application_point[0]}, {lc.application_point[1]}, {lc.application_point[2]})."
        )
    else:
        suggestions.append(f"{len(setup.loads)} load cases considered.")

    bc_types = [bc.bc_type for bc in setup.constraints]
    suggestions.append(
        f"Boundary conditions: {', '.join(bc_types)} "
        f"on faces {', '.join(bc.face_selector for bc in setup.constraints)}."
    )

    if volume_reduction_pct > 50:
        suggestions.append(
            "High volume reduction — consider using lattice infill or ribs "
            "in the removed regions for structural integrity."
        )
    elif volume_reduction_pct > 20:
        suggestions.append(
            "Moderate volume reduction — the optimized shape should be "
            "manufacturable with CNC milling or 3D printing."
        )

    return suggestions
