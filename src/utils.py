import numpy as np

def as_cartesian(rthetaphi):
    r, theta, phi = rthetaphi
    theta, phi = np.deg2rad([theta, phi])
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return [x,y,z]

def as_spherical(xyz):
    x, y, z = xyz
    r = np.sqrt(x*x + y*y + z*z)
    theta, phi = np.rad2deg([np.arccos(z/r), np.arctan2(y,x)])
    return [r, theta, phi]

def transform_coordinates(coordinates):
    return np.array([as_spherical(coord) for coord in coordinates])
