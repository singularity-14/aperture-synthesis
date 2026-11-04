"""apsynth - a minimal, self-validating aperture synthesis simulator.

Simulates how a radio interferometer samples the Fourier transform of the sky,
inverts those samples into a dirty image, and deconvolves it with Hogbom CLEAN.

The point is not to be a production imager - WSClean exists. The point is that
every step is short enough to read, and the package checks its own physics:
``tests/test_imaging.py`` asserts the identity ``dirty == sky (*) beam``, and
the demo measures how accurately CLEAN recovers known input flux densities.

Quick start::

    import astropy.units as u
    from apsynth import arrays, uvcoverage, sky, imaging, clean

    enu = arrays.y_array(n_per_arm=9, r_max=2000.0)
    xyz = uvcoverage.enu_to_xyz(enu, latitude=34 * u.deg)
    tracks = uvcoverage.uvw_tracks(
        uvcoverage.baseline_vectors(xyz),
        hour_angles=np.linspace(-4, 4, 120) * u.hourangle,
        declination=45 * u.deg,
        wavelength_m=uvcoverage.wavelength(1.4 * u.GHz),
    )
    u_coords, v_coords = uvcoverage.uv_points(tracks)

    model = sky.SkyModel(n_pix=256, cell=1 * u.arcsec)
    model.add_point(sky.PointSource(10 * u.arcsec, -5 * u.arcsec, flux=1.0))

    sampling = imaging.grid_samples(u_coords, v_coords, 256, 1 * u.arcsec)
    obs = imaging.invert(model.image, sampling, cell=1 * u.arcsec)
    result = clean.hogbom_clean(obs.dirty, obs.beam)
"""

from __future__ import annotations

from . import arrays, clean, imaging, metrics, plotting, sky, uvcoverage

__version__ = "0.1.0"

__all__ = [
    "arrays",
    "clean",
    "imaging",
    "metrics",
    "plotting",
    "sky",
    "uvcoverage",
    "__version__",
]
