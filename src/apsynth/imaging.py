"""Gridding, the dirty image, and the dirty beam.

The chain this module implements::

    sky (Jy/pixel)  --FFT-->  true visibilities V(u,v)
    uv samples      -------->  sampling function S(u,v)
    V * S           --iFFT-->  dirty image  (Jy/beam)
    S               --iFFT-->  dirty beam   (unit peak)

and the identity that makes deconvolution possible at all:

    dirty image = sky (*) dirty beam

That identity is asserted in ``tests/test_imaging.py``. If it ever fails, the
normalisation or the FFT convention is wrong and nothing downstream means
anything.

FFT convention
--------------
Everything here is in "centred" order: the zero spatial frequency and the field
centre both sit at index ``n_pix // 2``. The ``fftshift``/``ifftshift`` pairs in
:func:`_fft2c` and :func:`_ifft2c` convert to and from numpy's native ordering.

Arrays are indexed ``[row, col] = [m, l]`` for images and ``[v, u]`` for
visibility grids.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import astropy.units as u
import numpy as np

__all__ = ["Observation", "grid_samples", "invert", "uv_cell_size", "max_baseline_resolution"]


def _fft2c(a: np.ndarray) -> np.ndarray:
    """Centred forward 2D FFT."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a)))


def _ifft2c(a: np.ndarray) -> np.ndarray:
    """Centred inverse 2D FFT."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(a)))


def uv_cell_size(n_pix: int, cell) -> float:
    """Spacing of the uv grid, in wavelengths.

    Fourier duality: an image of ``n_pix`` cells of size ``cell`` radians
    corresponds to a uv grid of spacing ``1 / (n_pix * cell)`` extending to
    ``+-1 / (2 * cell)``.

    So: **a wider field of view means finer uv sampling, and smaller pixels
    mean the grid reaches longer baselines.** If your longest baseline falls
    outside the grid, you are silently throwing data away.
    """
    cell_rad = u.Quantity(cell).to_value(u.rad)
    return 1.0 / (n_pix * cell_rad)


def max_baseline_resolution(u_coords: np.ndarray, v_coords: np.ndarray) -> u.Quantity:
    """Rough synthesised beam size, ``theta ~ 1 / b_max``, as an angle.

    A sanity check, not a substitute for fitting the actual beam.
    """
    b_max = float(np.max(np.hypot(u_coords, v_coords)))
    if b_max <= 0:
        raise ValueError("no non-zero baselines")
    return (1.0 / b_max * u.rad).to(u.arcsec)


def grid_samples(
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    n_pix: int,
    cell,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Bin uv samples onto a regular grid to build the sampling function.

    Each sample is also entered at ``(-u, -v)``. That is not a trick to double
    the data: the sky is real, so ``V(-u,-v) = conj(V(u,v))`` and the conjugate
    point genuinely is measured by the same antenna pair. Including it is what
    makes the dirty image come out real rather than complex.

    Nearest-cell binning is used for clarity. Production imagers convolve each
    sample onto several cells with a prolate spheroidal kernel to suppress
    aliasing; that is a refinement, not a different idea.

    Parameters
    ----------
    u_coords, v_coords : 1D sample coordinates in wavelengths.
    n_pix : grid size, matching the image.
    cell : image cell size as a ``Quantity`` angle.
    weights : optional per-sample weight. Defaults to 1 (natural weighting).

    Returns
    -------
    (n_pix, n_pix) real sampling function, centred, indexed ``[v, u]``.
    """
    du = uv_cell_size(n_pix, cell)
    if weights is None:
        weights = np.ones_like(u_coords, dtype=float)

    uu = np.concatenate([u_coords, -u_coords])
    vv = np.concatenate([v_coords, -v_coords])
    ww = np.concatenate([weights, weights])

    col = np.rint(uu / du).astype(int) + n_pix // 2
    row = np.rint(vv / du).astype(int) + n_pix // 2

    inside = (row >= 0) & (row < n_pix) & (col >= 0) & (col < n_pix)
    n_dropped = int((~inside).sum())
    if n_dropped:
        # Loud on purpose: silently dropping long baselines degrades resolution
        # in a way that is very hard to spot in the final image.
        warnings.warn(
            f"{n_dropped} of {inside.size} uv samples fall outside the grid and "
            f"were dropped; decrease `cell` to reach longer baselines.",
            stacklevel=2,
        )

    sampling = np.zeros((n_pix, n_pix), dtype=float)
    np.add.at(sampling, (row[inside], col[inside]), ww[inside])
    return sampling


@dataclass
class Observation:
    """The result of observing a sky model with a given uv-coverage."""

    dirty: np.ndarray
    """Dirty image, Jy/beam. Equals ``sky (*) beam``."""

    beam: np.ndarray
    """Dirty beam (point spread function), normalised to unit peak."""

    sampling: np.ndarray
    """The sampling function S(u,v) that produced them."""

    visibilities: np.ndarray
    """Sampled complex visibilities on the uv grid, ``V * S``."""

    cell: u.Quantity
    """Image cell size."""

    @property
    def n_pix(self) -> int:
        return self.dirty.shape[0]

    @property
    def uv_fill_fraction(self) -> float:
        """Fraction of uv cells with at least one sample. A coverage summary."""
        return float((self.sampling > 0).sum() / self.sampling.size)


def invert(
    sky_image: np.ndarray,
    sampling: np.ndarray,
    cell,
    noise_jy: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Observation:
    """Forward-model an observation and invert it to a dirty image.

    Normalisation is chosen so the beam has unit peak and the identity
    ``dirty == sky (*) beam`` holds exactly (to floating point).

    With numpy's FFT convention ``ifft2(A)[0,0] == A.sum() / N**2``, so the
    un-normalised beam peaks at ``sampling.sum() / n_pix**2``. Dividing both the
    beam and the dirty image by that value fixes the beam peak at 1 and leaves
    the dirty image in Jy/beam.

    Parameters
    ----------
    sky_image : (n, n) sky brightness in Jy/pixel.
    sampling : (n, n) sampling function from :func:`grid_samples`.
    cell : image cell size as a ``Quantity`` angle.
    noise_jy : standard deviation of Gaussian thermal noise added to the real
        and imaginary part of each *sampled* visibility. Zero disables noise.
    rng : optional random generator, for reproducibility.

    Returns
    -------
    Observation
    """
    if sky_image.shape != sampling.shape:
        raise ValueError("sky image and sampling function must have the same shape")

    n_pix = sky_image.shape[0]
    norm = sampling.sum() / n_pix**2
    if norm <= 0:
        raise ValueError("sampling function is empty; no visibilities were gridded")

    true_vis = _fft2c(sky_image)
    sampled_vis = true_vis * sampling

    if noise_jy > 0:
        rng = rng or np.random.default_rng()
        noise = rng.normal(0.0, noise_jy, sampled_vis.shape) + 1j * rng.normal(
            0.0, noise_jy, sampled_vis.shape
        )
        # Each grid cell holds the *sum* of the samples that landed in it, so a
        # cell with k samples carries noise sigma*sqrt(k), not sigma.
        noise = noise * np.sqrt(sampling)
        # Force the noise Hermitian so the dirty image is real for the right
        # reason rather than because we threw the imaginary part away.
        # On a centred grid the conjugate of index c+d is c-d, i.e. index k maps
        # to N-k. Reversing gives N-1-k, so a roll of +1 completes the mapping.
        noise = 0.5 * (noise + np.conj(np.roll(noise[::-1, ::-1], (1, 1), axis=(0, 1))))
        sampled_vis = sampled_vis + noise

    dirty = _ifft2c(sampled_vis).real / norm
    beam = _ifft2c(sampling).real / norm

    return Observation(
        dirty=dirty,
        beam=beam,
        sampling=sampling,
        visibilities=sampled_vis,
        cell=u.Quantity(cell),
    )
