"""Azimuth rotations in the declared x=right,y=front convention."""
import numpy as np


def rotate_azimuth(unit_direction,degrees):
    u=np.asarray(unit_direction,dtype=np.float64)
    if u.shape!=(3,) or not np.isfinite(u).all() or not np.isclose(np.linalg.norm(u),1.,atol=1e-8) or not np.isfinite(degrees):
        raise ValueError('Require a finite unit3-vector and finite angle')
    a=np.deg2rad(degrees)
    return np.array([np.cos(a)*u[0]+np.sin(a)*u[1],-np.sin(a)*u[0]+np.cos(a)*u[1],u[2]])


def azimuth_degrees(unit_direction):
    u=np.asarray(unit_direction,dtype=np.float64)
    if u.shape!=(3,) or not np.isfinite(u).all():raise ValueError('Require a finite3-vector')
    return float(np.rad2deg(np.arctan2(u[0],u[1])))


def wrap_degrees(angle):return (float(angle)+180.)%360.-180.
