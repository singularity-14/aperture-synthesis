"""Shared fixtures.

A deliberately small configuration so the whole suite runs in a few seconds,
while keeping the synthesised beam sampled across several pixels - a beam
narrower than about three pixels cannot be Gaussian-fitted reliably, and the
clean-beam tests would fail for an uninteresting reason.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from apsynth import arrays, imaging, sky, uvcoverage

N_PIX = 128
CELL = 4 * u.arcsec
FREQUENCY = 1.4 * u.GHz
LATITUDE = 34 * u.deg
DECLINATION = 45 * u.deg


@pytest.fixture(scope="session")
def uv_sample():
    """u, v coordinates in wavelengths for a short synthesis run."""
    enu = arrays.y_array(n_per_arm=7, r_max=1200.0)
    xyz = uvcoverage.enu_to_xyz(enu, LATITUDE)
    baselines = uvcoverage.baseline_vectors(xyz)
    hour_angles = np.linspace(-3.0, 3.0, 60) * u.hourangle
    tracks = uvcoverage.uvw_tracks(
        baselines, hour_angles, DECLINATION, uvcoverage.wavelength(FREQUENCY)
    )
    return uvcoverage.uv_points(tracks)


@pytest.fixture(scope="session")
def sampling(uv_sample):
    u_coords, v_coords = uv_sample
    return imaging.grid_samples(u_coords, v_coords, N_PIX, CELL)


@pytest.fixture
def point_sources():
    return [
        sky.PointSource(l=0 * u.arcsec, m=0 * u.arcsec, flux=1.0),
        sky.PointSource(l=60 * u.arcsec, m=40 * u.arcsec, flux=0.5),
        sky.PointSource(l=-80 * u.arcsec, m=-60 * u.arcsec, flux=0.25),
    ]


@pytest.fixture
def sky_model(point_sources):
    model = sky.SkyModel(n_pix=N_PIX, cell=CELL)
    for source in point_sources:
        model.add_point(source)
    return model


@pytest.fixture
def observation(sky_model, sampling):
    return imaging.invert(sky_model.image, sampling, cell=CELL)
