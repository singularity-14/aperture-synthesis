"""Antenna array layouts.

Positions are returned in a local horizon frame, in metres:

    E = East, N = North, U = Up

This is the frame a surveyor would hand you. :func:`apsynth.uvcoverage.enu_to_xyz`
converts it to the equatorial frame the uvw formulae need.
"""

from __future__ import annotations

import numpy as np

__all__ = ["y_array", "east_west_array", "circular_array", "random_array"]


def y_array(
    n_per_arm: int = 9,
    r_max: float = 1000.0,
    power: float = 1.716,
    arm_angles_deg: tuple[float, ...] = (0.0, 120.0, 240.0),
) -> np.ndarray:
    """A VLA-like Y configuration.

    Three arms at 120 degrees, with antennas spaced along each arm by a power
    law ``r_i = r_max * ((i+1)/n)**power``. The power law is what gives the VLA
    its characteristic smooth, well-filled uv-coverage: equally spaced antennas
    would produce redundant baselines (many antenna pairs measuring the *same*
    Fourier component), which is wasteful if your goal is imaging.

    This is *VLA-like*, not the VLA. Real VLA antenna positions are published;
    use those if you need to reproduce a real observation.

    Parameters
    ----------
    n_per_arm : number of antennas on each arm.
    r_max : distance of the outermost antenna from the array centre, metres.
    power : power-law exponent for antenna spacing along an arm.
    arm_angles_deg : azimuth of each arm, degrees east of north.

    Returns
    -------
    (n_ant, 3) array of E, N, U positions in metres.
    """
    radii = r_max * ((np.arange(1, n_per_arm + 1)) / n_per_arm) ** power
    positions = []
    for angle_deg in arm_angles_deg:
        theta = np.deg2rad(angle_deg)
        for r in radii:
            positions.append([r * np.sin(theta), r * np.cos(theta), 0.0])
    return np.asarray(positions, dtype=float)


def east_west_array(n_ant: int = 8, spacing: float = 150.0) -> np.ndarray:
    """A linear east-west array.

    Instructive because its snapshot uv-coverage is a *line*, so a single
    snapshot cannot constrain structure in the perpendicular direction at all.
    Earth rotation then sweeps that line into a set of ellipses. Compare its
    snapshot and full-synthesis beams to see why earth rotation synthesis was
    such an important idea.
    """
    east = (np.arange(n_ant) - (n_ant - 1) / 2) * spacing
    return np.column_stack([east, np.zeros(n_ant), np.zeros(n_ant)])


def circular_array(n_ant: int = 16, radius: float = 500.0) -> np.ndarray:
    """Antennas evenly spaced on a circle."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_ant, endpoint=False)
    return np.column_stack(
        [radius * np.sin(theta), radius * np.cos(theta), np.zeros(n_ant)]
    )


def random_array(
    n_ant: int = 24, radius: float = 600.0, seed: int | None = 42
) -> np.ndarray:
    """Antennas at uniformly random positions inside a disc.

    Randomness is a surprisingly good imaging layout: it produces almost no
    redundant baselines. Real arrays compromise, because cable routing, terrain
    and maintenance access all cost money.
    """
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n_ant))
    theta = rng.uniform(0.0, 2.0 * np.pi, n_ant)
    return np.column_stack([r * np.sin(theta), r * np.cos(theta), np.zeros(n_ant)])
