"""The physics the rest of the package depends on.

If ``test_dirty_image_is_sky_convolved_with_beam`` fails, nothing downstream
means anything: CLEAN is built entirely on the assumption that the dirty image
is a convolution it can undo.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from apsynth import imaging
from apsynth.clean import convolve_fft

from .conftest import CELL, N_PIX


def test_dirty_image_is_sky_convolved_with_beam(sky_model, observation):
    """The identity that makes deconvolution a well-posed idea.

    ``dirty == sky (*) dirty_beam``, exactly, to floating-point precision.
    """
    expected = convolve_fft(sky_model.image, observation.beam)
    np.testing.assert_allclose(observation.dirty, expected, atol=1e-10, rtol=0)


def test_beam_has_unit_peak_at_centre(observation):
    """The beam is normalised so the dirty image reads directly in Jy/beam."""
    centre = N_PIX // 2
    assert observation.beam[centre, centre] == pytest.approx(1.0, abs=1e-10)
    assert np.argmax(observation.beam) == centre * N_PIX + centre


def test_point_source_peak_equals_its_flux(sampling):
    """A single unresolved source peaks at its flux density in Jy/beam.

    Because the beam has unit peak and a point source is one pixel of Jy, the
    dirty image peak must equal the input flux. This is the unit convention
    working as intended.
    """
    from apsynth import sky as sky_module

    model = sky_module.SkyModel(n_pix=N_PIX, cell=CELL)
    model.add_point(sky_module.PointSource(0 * u.arcsec, 0 * u.arcsec, flux=2.5))
    obs = imaging.invert(model.image, sampling, cell=CELL)
    assert obs.dirty.max() == pytest.approx(2.5, rel=1e-9)


def test_dirty_image_is_real(observation):
    """No imaginary leakage: the sampling function is Hermitian by construction."""
    assert np.isrealobj(observation.dirty)
    assert np.all(np.isfinite(observation.dirty))


def test_sampling_function_is_hermitian(sampling):
    """``S(u,v) == S(-u,-v)`` because every sample is entered with its conjugate.

    On a centred even grid the conjugate of index ``c+d`` sits at ``c-d``,
    i.e. index ``k`` maps to ``N-k``, which is ``[::-1]`` followed by a roll of 1.
    """
    flipped = np.roll(sampling[::-1, ::-1], (1, 1), axis=(0, 1))
    np.testing.assert_allclose(sampling, flipped)


def test_uv_cell_size_is_fourier_dual_of_the_image():
    """``du = 1 / (n_pix * cell)`` - the Fourier duality the grid rests on."""
    cell = 2 * u.arcsec
    du = imaging.uv_cell_size(256, cell)
    assert du == pytest.approx(1.0 / (256 * cell.to_value(u.rad)))


def test_samples_outside_the_grid_warn_rather_than_vanish(uv_sample):
    """Silently dropping long baselines would quietly destroy resolution."""
    u_coords, v_coords = uv_sample
    with pytest.warns(UserWarning, match="outside the grid"):
        imaging.grid_samples(u_coords, v_coords, N_PIX, 40 * u.arcsec)


def test_empty_sampling_is_rejected(sky_model):
    with pytest.raises(ValueError, match="empty"):
        imaging.invert(sky_model.image, np.zeros((N_PIX, N_PIX)), cell=CELL)


def test_noise_keeps_the_dirty_image_real_and_raises_the_floor(sky_model, sampling):
    """Thermal noise should raise the noise floor without making the image complex."""
    rng = np.random.default_rng(0)
    quiet = imaging.invert(sky_model.image, sampling, cell=CELL)
    noisy = imaging.invert(sky_model.image, sampling, cell=CELL, noise_jy=0.01, rng=rng)

    assert np.isrealobj(noisy.dirty)
    corner = (slice(0, 16), slice(0, 16))  # source-free region
    assert np.std(noisy.dirty[corner]) > np.std(quiet.dirty[corner])


def test_uv_fill_fraction_is_a_small_positive_number(observation):
    """Real coverage is sparse - that sparsity is the entire problem."""
    assert 0.0 < observation.uv_fill_fraction < 0.5
