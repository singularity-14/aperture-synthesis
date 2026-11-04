"""CLEAN behaviour, and the flux-recovery check that decides whether it works."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from apsynth import clean, imaging, metrics
from apsynth import sky as sky_module

from .conftest import CELL, N_PIX


def test_clean_recovers_input_fluxes(sky_model, observation, point_sources):
    """The headline check: does the flux that went in come back out?

    Everything else about an image is decoration if the flux scale is wrong.

    This field is noiseless, so CLEAN is run deep (``threshold_sigma=0.05``).
    On real data that would be reckless - you would be fitting noise - but here
    it isolates the question this test is actually asking: is the *algorithm*
    flux-conserving, independent of where you choose to stop it? The separate
    effect of stopping early is pinned down in
    :func:`test_shallow_cleaning_leaves_flux_behind`.
    """
    result = clean.hogbom_clean(
        observation.dirty, observation.beam, gain=0.1, max_iter=20000, threshold_sigma=0.05
    )
    comparisons = metrics.compare_fluxes(result.components, point_sources, CELL)

    for comparison in comparisons:
        assert abs(comparison.percent_error) < 1.0, (
            f"{comparison.name}: in {comparison.input_flux:.3f} Jy, "
            f"out {comparison.recovered_flux:.3f} Jy"
        )

    total_in = sum(c.input_flux for c in comparisons)
    total_out = sum(c.recovered_flux for c in comparisons)
    assert total_out == pytest.approx(total_in, rel=0.01)


def test_clean_components_are_not_scattered_off_the_source(observation, point_sources):
    """All of a point source's flux lands in essentially one pixel.

    Widening the summing aperture must not change the measured flux. If it
    does, components are leaking into the surroundings and the flux-recovery
    numbers would be an artefact of aperture choice rather than a real result.
    """
    result = clean.hogbom_clean(
        observation.dirty, observation.beam, max_iter=20000, threshold_sigma=0.05
    )
    tight = metrics.compare_fluxes(result.components, point_sources, CELL, aperture_pix=1)
    wide = metrics.compare_fluxes(result.components, point_sources, CELL, aperture_pix=8)

    for narrow, broad in zip(tight, wide):
        assert narrow.recovered_flux == pytest.approx(broad.recovered_flux, rel=1e-9)


def test_shallow_cleaning_leaves_flux_behind(observation, point_sources):
    """CLEAN stopped early under-reports flux, and faint sources suffer most.

    The loop halts with roughly ``threshold`` of flux still under each source.
    That deficit is *absolute*, so as a *fraction* it hurts a faint source more
    than a bright one - a real and frequently overlooked photometric bias, not
    a defect in this implementation.
    """
    deep = clean.hogbom_clean(
        observation.dirty, observation.beam, max_iter=20000, threshold_sigma=0.05
    )
    shallow = clean.hogbom_clean(
        observation.dirty, observation.beam, max_iter=20000, threshold_sigma=3.0
    )

    assert shallow.model_flux < deep.model_flux

    errors = [
        abs(c.percent_error)
        for c in metrics.compare_fluxes(shallow.components, point_sources, CELL)
    ]
    fluxes = [s.flux for s in point_sources]
    brightest = errors[int(np.argmax(fluxes))]
    faintest = errors[int(np.argmin(fluxes))]
    assert faintest > brightest, "fractional flux loss should be worse for faint sources"


def test_clean_reduces_the_residual_peak(observation):
    result = clean.hogbom_clean(observation.dirty, observation.beam, max_iter=500)
    assert result.peak_history[-1] < result.peak_history[0]
    assert np.max(np.abs(result.residual)) < np.max(np.abs(observation.dirty))


def test_clean_improves_dynamic_range(observation):
    """Deconvolution buys back the dynamic range the sidelobes were costing."""
    result = clean.hogbom_clean(observation.dirty, observation.beam, max_iter=3000)
    before = metrics.dynamic_range(observation.dirty, observation.dirty)
    after = metrics.dynamic_range(result.restored, result.residual)
    assert after > before


def test_clean_converges_on_a_clean_field(sampling):
    """An empty sky should produce no components and converge immediately."""
    empty = np.zeros((N_PIX, N_PIX))
    obs = imaging.invert(empty, sampling, cell=CELL)
    result = clean.hogbom_clean(obs.dirty, obs.beam)
    assert result.converged
    assert result.model_flux == pytest.approx(0.0, abs=1e-12)


def test_clean_components_land_on_the_source(sampling):
    """A single source puts its components where the source actually is."""
    model = sky_module.SkyModel(n_pix=N_PIX, cell=CELL)
    model.add_point(sky_module.PointSource(40 * u.arcsec, -20 * u.arcsec, flux=1.0))
    obs = imaging.invert(model.image, sampling, cell=CELL)
    result = clean.hogbom_clean(obs.dirty, obs.beam, max_iter=2000)

    peak_index = np.unravel_index(np.argmax(result.components), result.components.shape)
    expected = np.unravel_index(np.argmax(model.image), model.image.shape)
    assert peak_index == expected


def test_restored_image_equals_model_convolved_plus_residual(observation):
    """The definition of the restored image, asserted rather than assumed."""
    result = clean.hogbom_clean(observation.dirty, observation.beam, max_iter=400)
    expected = clean.convolve_fft(result.components, result.clean_beam) + result.residual
    np.testing.assert_allclose(result.restored, expected, atol=1e-10)


def test_clean_beam_is_narrower_than_the_field_and_well_formed(observation):
    _, params = clean.fit_clean_beam(observation.beam)
    assert params["bmaj_pix"] >= params["bmin_pix"] > 0
    assert params["bmaj_pix"] < N_PIX / 4
    assert 0.0 <= params["bpa_deg"] < 180.0


def test_clean_beam_is_wider_than_a_pixel(observation):
    """A restoring beam narrower than a pixel would mean the fit failed."""
    _, params = clean.fit_clean_beam(observation.beam)
    assert params["bmin_pix"] > 1.0


def test_higher_gain_needs_fewer_iterations(observation):
    """Loop gain trades speed against stability, as advertised."""
    slow = clean.hogbom_clean(observation.dirty, observation.beam, gain=0.05, max_iter=5000)
    fast = clean.hogbom_clean(observation.dirty, observation.beam, gain=0.3, max_iter=5000)
    assert fast.n_iter < slow.n_iter


def test_invalid_gain_is_rejected(observation):
    with pytest.raises(ValueError, match="gain"):
        clean.hogbom_clean(observation.dirty, observation.beam, gain=0.0)


def test_mismatched_shapes_are_rejected(observation):
    with pytest.raises(ValueError, match="same shape"):
        clean.hogbom_clean(observation.dirty, observation.beam[:-2, :-2])


def test_beam_area_matches_the_analytic_gaussian_formula():
    """``pi * bmaj * bmin / (4 ln 2)`` - the Jy/beam to Jy conversion factor."""
    assert metrics.beam_area_pixels(4.0, 2.0) == pytest.approx(
        np.pi * 8.0 / (4 * np.log(2))
    )


def test_robust_rms_ignores_bright_sources():
    """MAD-based noise estimation must not be dragged up by real signal."""
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 1.0, (256, 256))
    with_source = noise.copy()
    with_source[100:104, 100:104] = 500.0

    assert metrics.robust_rms(with_source) == pytest.approx(1.0, rel=0.1)
    assert np.std(with_source) > 3.0  # the non-robust estimate is wrecked
