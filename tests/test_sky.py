"""Sky model construction and unit bookkeeping."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from apsynth import sky as sky_module


def test_point_source_flux_lands_in_one_pixel():
    model = sky_module.SkyModel(n_pix=64, cell=1 * u.arcsec)
    model.add_point(sky_module.PointSource(0 * u.arcsec, 0 * u.arcsec, flux=3.0))
    assert model.total_flux == pytest.approx(3.0)
    assert np.count_nonzero(model.image) == 1
    assert model.image[32, 32] == pytest.approx(3.0)


def test_source_position_maps_to_the_right_pixel():
    """l increases along columns (East), m along rows (North)."""
    model = sky_module.SkyModel(n_pix=64, cell=2 * u.arcsec)
    model.add_point(sky_module.PointSource(10 * u.arcsec, -6 * u.arcsec, flux=1.0))
    row, col = np.unravel_index(np.argmax(model.image), model.image.shape)
    assert col == 32 + 5  # +10" at 2"/pixel
    assert row == 32 - 3  # -6"  at 2"/pixel


def test_gaussian_source_conserves_total_flux():
    """A resolved source spreads over many pixels but keeps its total flux."""
    model = sky_module.SkyModel(n_pix=128, cell=1 * u.arcsec)
    model.add_gaussian(
        sky_module.GaussianSource(0 * u.arcsec, 0 * u.arcsec, flux=2.0, fwhm=8 * u.arcsec)
    )
    assert model.total_flux == pytest.approx(2.0, rel=1e-6)
    assert np.count_nonzero(model.image) > 50


def test_fluxes_accumulate():
    model = sky_module.SkyModel(n_pix=64, cell=1 * u.arcsec)
    model.add_point(sky_module.PointSource(0 * u.arcsec, 0 * u.arcsec, flux=1.0))
    model.add_point(sky_module.PointSource(5 * u.arcsec, 5 * u.arcsec, flux=2.0))
    assert model.total_flux == pytest.approx(3.0)


def test_source_outside_the_field_is_rejected():
    """Better a loud error than a source silently wrapped to the other side."""
    model = sky_module.SkyModel(n_pix=64, cell=1 * u.arcsec)
    with pytest.raises(ValueError, match="outside the image"):
        model.add_point(sky_module.PointSource(500 * u.arcsec, 0 * u.arcsec, flux=1.0))


def test_odd_image_size_is_rejected():
    """An odd grid puts the phase centre between pixels and breaks the FFT convention."""
    with pytest.raises(ValueError, match="even"):
        sky_module.SkyModel(n_pix=65, cell=1 * u.arcsec)


def test_units_are_enforced():
    """Passing a bare number where an angle belongs must fail, not silently work."""
    model = sky_module.SkyModel(n_pix=64, cell=1 * u.arcsec)
    with pytest.raises((u.UnitsError, u.UnitConversionError, TypeError, AttributeError)):
        model.add_point(sky_module.PointSource(10, 0 * u.arcsec, flux=1.0))


def test_pixel_coords_centre_is_zero():
    l_grid, m_grid = sky_module.pixel_coords(64, 1 * u.arcsec)
    assert l_grid[32, 32] == pytest.approx(0.0)
    assert m_grid[32, 32] == pytest.approx(0.0)
    assert l_grid[32, 33] > 0  # l increases with column
    assert m_grid[33, 32] > 0  # m increases with row
