"""Physical properties computation.

Computes mass, center of gravity, moments of inertia from B-Rep geometry.
Uses OpenCascade's GProp facilities for exact computation on analytic geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopoDS import TopoDS_Shape

from next3d.core.schema import Vec3


@dataclass(frozen=True)
class PhysicalProperties:
    """Physical properties of a solid body."""

    volume: float  # mm³
    surface_area: float  # mm²
    center_of_gravity: Vec3
    mass: float  # grams (given density)
    density: float  # g/mm³

    # Principal moments of inertia (g·mm²)
    ixx: float
    iyy: float
    izz: float

    # Products of inertia
    ixy: float
    ixz: float
    iyz: float

    def to_dict(self) -> dict:
        return {
            "volume_mm3": round(self.volume, 4),
            "surface_area_mm2": round(self.surface_area, 4),
            "center_of_gravity": {
                "x": round(self.center_of_gravity.x, 6),
                "y": round(self.center_of_gravity.y, 6),
                "z": round(self.center_of_gravity.z, 6),
            },
            "mass_grams": round(self.mass, 4),
            "density_g_per_mm3": self.density,
            "moments_of_inertia": {
                "Ixx": round(self.ixx, 4),
                "Iyy": round(self.iyy, 4),
                "Izz": round(self.izz, 4),
            },
            "products_of_inertia": {
                "Ixy": round(self.ixy, 4),
                "Ixz": round(self.ixz, 4),
                "Iyz": round(self.iyz, 4),
            },
        }


# Common material densities (g/mm³)
MATERIALS = {
    "steel": 0.00785,
    "aluminum": 0.0027,
    "titanium": 0.00451,
    "brass": 0.0085,
    "copper": 0.00896,
    "nylon": 0.00114,
    "abs": 0.00105,
    "pla": 0.00125,
}


def compute_physical_properties(
    shape: TopoDS_Shape,
    density: float = 0.00785,  # steel by default
) -> PhysicalProperties:
    """Compute physical properties of a shape.

    Args:
        shape: The TopoDS_Shape (solid).
        density: Material density in g/mm³. Default is steel (7.85 g/cm³).

    Returns:
        PhysicalProperties with volume, mass, CoG, inertia.
    """
    # Volume properties
    vol_props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, vol_props)
    volume = vol_props.Mass()  # "Mass" in volume props = volume
    cog = vol_props.CentreOfMass()

    # Surface area
    surf_props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, surf_props)
    surface_area = surf_props.Mass()  # "Mass" in surface props = area

    # Mass
    mass = volume * density

    # Moments of inertia about the center of gravity
    mat = vol_props.MatrixOfInertia()
    # Scale by density to get mass-based inertia
    ixx = mat.Value(1, 1) * density
    iyy = mat.Value(2, 2) * density
    izz = mat.Value(3, 3) * density
    ixy = mat.Value(1, 2) * density
    ixz = mat.Value(1, 3) * density
    iyz = mat.Value(2, 3) * density

    return PhysicalProperties(
        volume=volume,
        surface_area=surface_area,
        center_of_gravity=Vec3(x=cog.X(), y=cog.Y(), z=cog.Z()),
        mass=mass,
        density=density,
        ixx=ixx,
        iyy=iyy,
        izz=izz,
        ixy=ixy,
        ixz=ixz,
        iyz=iyz,
    )
