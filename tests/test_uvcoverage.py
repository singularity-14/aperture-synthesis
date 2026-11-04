"""Geometry checks on baselines and the uvw rotation."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from apsynth import arrays, uvcoverage


def test_baseline_count_is_n_choose_2():
    enu = arrays.y_array(n_per_arm=5)
    assert len(enu) == 15
    assert len(uvcoverage.baseline_vectors(enu)) == 15 * 14 // 2


def test_enu_to_xyz_preserves_length():
    """A frame rotation cannot change the length of a vector."""
    enu = arrays.random_array(n_ant=12, radius=800.0, seed=1)
    xyz = uvcoverage.enu_to_xyz(enu, 34 * u.deg)
    np.testing.assert_allclose(
        np.linalg.norm(enu, axis=1), np.linalg.norm(xyz, axis=1), rtol=1e-12
    )


def test_enu_to_xyz_at_the_north_pole():
    """At latitude 90 degrees, Up points at the NCP and North points to -X."""
    enu = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    xyz = uvcoverage.enu_to_xyz(enu, 90 * u.deg)
    np.testing.assert_allclose(xyz[0], [0.0, 1.0, 0.0], atol=1e-12)  # East -> +Y
    np.testing.assert_allclose(xyz[1], [-1.0, 0.0, 0.0], atol=1e-12)  # North -> -X
    np.testing.assert_allclose(xyz[2], [0.0, 0.0, 1.0], atol=1e-12)  # Up -> +Z (NCP)


def test_uvw_rotation_preserves_baseline_length():
    """uvw is a rotation of the baseline, so |uvw| == |b| / lambda."""
    enu = arrays.y_array(n_per_arm=4, r_max=900.0)
    xyz = uvcoverage.enu_to_xyz(enu, 34 * u.deg)
    baselines = uvcoverage.baseline_vectors(xyz)
    lam = uvcoverage.wavelength(1.4 * u.GHz)

    tracks = uvcoverage.uvw_tracks(
        baselines, np.linspace(-4, 4, 17) * u.hourangle, 40 * u.deg, lam
    )
    lengths = np.linalg.norm(tracks, axis=-1)
    expected = np.broadcast_to(
        (np.linalg.norm(baselines, axis=1) / lam)[:, None], lengths.shape
    )

    np.testing.assert_allclose(lengths, expected, rtol=1e-12)


def test_uvw_at_zero_hour_angle_and_the_pole():
    """At H=0, dec=90: u=Y, v=-X, w=Z, straight from the rotation matrix."""
    baselines = np.array([[3.0, 5.0, 7.0]])
    tracks = uvcoverage.uvw_tracks(baselines, 0 * u.hourangle, 90 * u.deg, wavelength_m=1.0)
    np.testing.assert_allclose(tracks[0, 0], [5.0, -3.0, 7.0], atol=1e-12)


def test_earth_rotation_increases_uv_coverage():
    """The whole point of an 8-hour track rather than a snapshot."""
    enu = arrays.y_array(n_per_arm=6, r_max=1000.0)
    xyz = uvcoverage.enu_to_xyz(enu, 34 * u.deg)
    baselines = uvcoverage.baseline_vectors(xyz)
    lam = uvcoverage.wavelength(1.4 * u.GHz)

    snapshot = uvcoverage.uvw_tracks(baselines, 0 * u.hourangle, 45 * u.deg, lam)
    synthesis = uvcoverage.uvw_tracks(
        baselines, np.linspace(-4, 4, 200) * u.hourangle, 45 * u.deg, lam
    )

    assert synthesis.reshape(-1, 3).shape[0] == 200 * snapshot.shape[0]
    # And the samples really do move, rather than being 200 copies.
    spread = np.ptp(synthesis[:, :, 0], axis=1)
    assert np.median(spread) > 0


def test_wavelength_conversion():
    assert uvcoverage.wavelength(1.4 * u.GHz) == pytest.approx(0.2141, rel=1e-3)
    assert uvcoverage.wavelength(300 * u.MHz) == pytest.approx(0.9993, rel=1e-3)


@pytest.mark.parametrize(
    "layout",
    [
        arrays.y_array(),
        arrays.east_west_array(),
        arrays.circular_array(),
        arrays.random_array(),
    ],
)
def test_layouts_are_planar_enu_arrays(layout):
    assert layout.ndim == 2
    assert layout.shape[1] == 3
    assert np.allclose(layout[:, 2], 0.0)  # all flat sites for now
