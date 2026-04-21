import pyvista as pv
import pyacvd
import numpy as np
from scipy import stats
import random
from spherical_kde import SphericalKDE

def resample(pvmesh, n_points):
    clus = pyacvd.Clustering(pvmesh)
    clus.subdivide(3)
    clus.cluster(n_points)
    remesh = clus.create_mesh()
    return remesh

def nvd(pvmesh):
    xyz = np.transpose(pvmesh.point_normals)
    kde = stats.gaussian_kde(xyz)
    density = kde(xyz)
    return density


def spherical_nvd(pvmesh):
    xyz = pvmesh.points  # Use mesh points instead of normals
    # Convert Cartesian coordinates to spherical (radians)
    r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
    theta = np.arctan2(np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2), xyz[:, 2])  # polar angle
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])  # azimuthal angle

    # Initialize and compute the spherical KDE
    skde = SphericalKDE(phi, theta)
    density = skde(phi, theta)
    return density


def mesh_to_nvd(meshpath, Von_Misses_Fisher = False, n_points=None, clip=False, rotations=False, translations=False,
                scaling=False, max_rotation=10, max_translation=10, max_scaling=0.1):
    """
    Convert mesh to NVD.

    Parameters:
    - meshpath: Path to the mesh file.
    - Von_Misses_Fisher: compute spherical kde using von Misses-Fisher distribution (non-Eucl. data).
    - n_points: Number of points for resampling.
    - clip: If True, clip the mesh.
    - rotations: If True, apply random rotations.
    - translations: If True, apply random translations.
    - scaling: If True, apply uniform scaling.
    - max_rotation: Maximum rotation in degrees.
    - max_translation: Maximum translation along each axis.
    - max_scaling: Maximum scaling factor.

    Returns:
    - nv_density: NV density of the mesh.
    """

    mesh = pv.read(meshpath)

    if rotations:
        # Generate random rotation angles within [-max_rotation, max_rotation]
        rotation_x = random.uniform(-max_rotation, max_rotation)
        rotation_y = random.uniform(-max_rotation, max_rotation)
        rotation_z = random.uniform(-max_rotation, max_rotation)

        mesh.rotate_x(rotation_x)
        mesh.rotate_y(rotation_y)
        mesh.rotate_z(rotation_z)

    if translations:
        # Generate random translation values within [-max_translation, max_translation]
        translation_x = random.uniform(-max_translation, max_translation)
        translation_y = random.uniform(-max_translation, max_translation)
        translation_z = random.uniform(-max_translation, max_translation)

        mesh.translate([translation_x, translation_y, translation_z])

    if scaling:
        # Generate random scaling factor within [1-max_scaling, 1+max_scaling]
        scaling_factor = random.uniform(1 - max_scaling, 1 + max_scaling)
        mesh.points *= scaling_factor

    if n_points:
        mesh = resample(mesh, n_points)

    if clip:
        mesh = mesh.clip(normal='y', invert=False)

    if Von_Misses_Fisher:
        nv_density = spherical_nvd(mesh)
    else:
        nv_density = nvd(mesh)

    return nv_density

