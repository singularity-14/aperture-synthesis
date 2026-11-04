"""Baselines and uv-coverage, including earth rotation synthesis.

The central fact of interferometry: **one pair of antennas measures one complex
Fourier component of the sky brightness distribution.** Which component depends
on the vector between the antennas as projected onto the plane perpendicular to
the direction you are looking.

Because the Earth turns, that projection changes through the night, so a fixed
pair of antennas traces out an arc through the Fourier plane. That is earth
rotation synthesis, and it is why an 8-hour observation images far better than
a snapshot.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.constants import c as speed_of_light

__all__ = [
    "enu_to_xyz",
    "baseline_vectors",
    "uvw_tracks",
    "uv_points",
    "wavelength",
]


def wavelength(frequency) -> float:
    """Wavelength in metres from a frequency ``Quantity`` (e.g. ``1.4 * u.GHz``)."""
    freq = u.Quantity(frequency).to(u.Hz)
    return float((speed_of_light / freq).to_value(u.m))


def enu_to_xyz(enu: np.ndarray, latitude) -> np.ndarray:
    """Convert local East-North-Up positions to the equatorial XYZ frame.

    The equatorial frame used for uvw calculations is defined as:

    * X points toward the meridian at the celestial equator (hour angle 0, dec 0)
    * Y points toward hour angle -6h at the celestial equator
    * Z points toward the North Celestial Pole

    ::

        X = -N sin(lat) + U cos(lat)
        Y =  E
        Z =  N cos(lat) + U sin(lat)

    Parameters
    ----------
    enu : (n_ant, 3) positions in metres.
    latitude : observatory latitude as a ``Quantity`` (e.g. ``34 * u.deg``).

    Returns
    -------
    (n_ant, 3) positions in metres in the equatorial frame.
    """
    lat = u.Quantity(latitude).to_value(u.rad)
    east, north, up = enu[:, 0], enu[:, 1], enu[:, 2]
    x = -north * np.sin(lat) + up * np.cos(lat)
    y = east
    z = north * np.cos(lat) + up * np.sin(lat)
    return np.column_stack([x, y, z])


def baseline_vectors(xyz: np.ndarray) -> np.ndarray:
    """All unique antenna-pair difference vectors.

    For ``n`` antennas there are ``n(n-1)/2`` pairs. Only one of each pair is
    returned: the conjugate baseline measures the complex conjugate visibility
    and carries no independent information, so it is added later at the
    gridding step rather than being tracked here.

    Parameters
    ----------
    xyz : (n_ant, 3) equatorial positions in metres.

    Returns
    -------
    (n_baseline, 3) baseline vectors in metres.
    """
    i, j = np.triu_indices(len(xyz), k=1)
    return xyz[j] - xyz[i]


def uvw_tracks(
    baselines_m: np.ndarray,
    hour_angles,
    declination,
    wavelength_m: float,
) -> np.ndarray:
    """Project baselines into uvw coordinates over a range of hour angles.

    The standard rotation (Thompson, Moran & Swenson eq. 4.1)::

        [u]   [  sin H         cos H        0     ] [X]
        [v] = [ -sin d cos H   sin d sin H  cos d ] [Y]
        [w]   [  cos d cos H  -cos d sin H  sin d ] [Z]

    where ``H`` is hour angle and ``d`` is the declination of the field centre.

    ``u`` and ``v`` are the coordinates in the plane perpendicular to the
    pointing direction; ``w`` is along it. This code ignores ``w`` when imaging
    (the narrow-field approximation) - see the README for when that breaks.

    Parameters
    ----------
    baselines_m : (n_baseline, 3) baseline vectors in metres, equatorial frame.
    hour_angles : ``Quantity`` array of hour angles (e.g. ``np.linspace(-4, 4, 100) * u.hourangle``).
    declination : ``Quantity`` declination of the field centre.
    wavelength_m : observing wavelength in metres.

    Returns
    -------
    (n_baseline, n_time, 3) array of u, v, w **in units of wavelengths**.
    """
    ha = np.atleast_1d(u.Quantity(hour_angles).to_value(u.rad))
    dec = u.Quantity(declination).to_value(u.rad)

    # Baselines in wavelengths: this is what makes u,v dimensionless and
    # directly comparable to spatial frequency on the sky.
    bl = np.asarray(baselines_m, dtype=float) / wavelength_m
    bx, by, bz = bl[:, 0, None], bl[:, 1, None], bl[:, 2, None]

    sin_h, cos_h = np.sin(ha)[None, :], np.cos(ha)[None, :]
    sin_d, cos_d = np.sin(dec), np.cos(dec)

    uu = sin_h * bx + cos_h * by
    vv = -sin_d * cos_h * bx + sin_d * sin_h * by + cos_d * bz
    ww = cos_d * cos_h * bx - cos_d * sin_h * by + sin_d * bz

    return np.stack([uu, vv, ww], axis=-1)


def uv_points(tracks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ``(n_baseline, n_time, 3)`` uvw tracks into 1D u and v arrays."""
    flat = tracks.reshape(-1, 3)
    return flat[:, 0], flat[:, 1]
