"""Hogbom CLEAN deconvolution.

CLEAN (Hogbom 1974) is the algorithm radio astronomy has used for fifty years.
The idea is simple enough to state in one sentence: **assume the sky is a small
number of point sources, then repeatedly find the brightest peak in the
residual image and subtract a scaled copy of the dirty beam centred on it.**

Machine learning people will recognise this immediately: it is matching pursuit
/ greedy sparse deconvolution, published two decades before those names existed.
The sparsity prior is "the sky is mostly empty", which is true for fields of
compact sources and false for extended emission - which is exactly where CLEAN
struggles and why variants like multi-scale CLEAN exist.

The output is deliberately *not* the raw component list. The components have
infinite resolution, which is a lie: the data only constrain the sky up to the
resolution of the longest baseline. So the components are blurred with a
"clean beam" - a Gaussian fitted to the dirty beam's main lobe - and the
residuals are added back, to keep an honest record of what was not deconvolved.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from scipy.optimize import curve_fit

__all__ = ["CleanResult", "hogbom_clean", "fit_clean_beam", "convolve_fft"]


def convolve_fft(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Circular convolution of two centred, equal-shaped arrays."""
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.fft2(np.fft.ifftshift(image)) * np.fft.fft2(np.fft.ifftshift(kernel))
        )
    ).real


def _elliptical_gaussian(
    coords: tuple[np.ndarray, np.ndarray],
    sigma_x: float,
    sigma_y: float,
    theta: float,
) -> np.ndarray:
    """Unit-amplitude elliptical Gaussian centred at the origin."""
    x, y = coords
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)
    return np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))


def fit_clean_beam(
    dirty_beam: np.ndarray, fit_level: float = 0.35, box: int | None = None
) -> tuple[np.ndarray, dict]:
    """Fit an elliptical Gaussian to the main lobe of the dirty beam.

    Only the central region is fitted: the dirty beam's sidelobes are precisely
    the part we are trying to remove, so including them would corrupt the fit.

    Parameters
    ----------
    dirty_beam : centred dirty beam, unit peak.
    fit_level : only pixels above this fraction of the peak are fitted.
    box : half-width in pixels of the region considered. Defaults to 1/8 of
        the image, which comfortably contains the main lobe for sane coverage.

    Returns
    -------
    clean_beam : (n, n) unit-peak Gaussian on the full grid.
    params : dict with ``bmaj_pix``, ``bmin_pix`` (FWHM in pixels) and
        ``bpa_deg`` (position angle of the major axis, degrees).
    """
    n_pix = dirty_beam.shape[0]
    centre = n_pix // 2
    box = box or max(8, n_pix // 8)

    lo, hi = centre - box, centre + box + 1
    patch = dirty_beam[lo:hi, lo:hi]
    axis = np.arange(-box, box + 1, dtype=float)
    x_patch, y_patch = np.meshgrid(axis, axis, indexing="xy")

    mask = patch > fit_level
    if mask.sum() < 6:
        raise RuntimeError(
            "dirty beam main lobe is too small to fit; the uv-coverage may be "
            "degenerate or `cell` too coarse"
        )

    # Initial guess: an equivalent-area circular Gaussian from the pixel count
    # above the fit level.
    sigma_guess = max(np.sqrt(mask.sum() / np.pi) / 1.5, 0.8)
    try:
        popt, _ = curve_fit(
            _elliptical_gaussian,
            (x_patch[mask], y_patch[mask]),
            patch[mask],
            p0=[sigma_guess, sigma_guess, 0.0],
            maxfev=20000,
        )
        sigma_x, sigma_y, theta = popt
        sigma_x, sigma_y = abs(sigma_x), abs(sigma_y)
    except RuntimeError:  # pragma: no cover - fallback for pathological beams
        sigma_x = sigma_y = sigma_guess
        theta = 0.0

    full_axis = np.arange(n_pix, dtype=float) - centre
    x_full, y_full = np.meshgrid(full_axis, full_axis, indexing="xy")
    clean_beam = _elliptical_gaussian((x_full, y_full), sigma_x, sigma_y, theta)

    to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
    bmaj, bmin = sorted([sigma_x * to_fwhm, sigma_y * to_fwhm], reverse=True)
    params = {
        "bmaj_pix": float(bmaj),
        "bmin_pix": float(bmin),
        "bpa_deg": float(np.rad2deg(theta) % 180.0),
    }
    return clean_beam, params


@dataclass
class CleanResult:
    """Everything CLEAN produced, including the things people forget to keep."""

    components: np.ndarray
    """Delta-function component image, Jy. The model CLEAN actually fitted."""

    residual: np.ndarray
    """What was left in the dirty image, Jy/beam. Its RMS is your noise estimate."""

    restored: np.ndarray
    """``components (*) clean_beam + residual``, Jy/beam. The image you publish."""

    clean_beam: np.ndarray
    """The Gaussian restoring beam, unit peak."""

    beam_params: dict
    """``bmaj_pix``, ``bmin_pix``, ``bpa_deg`` of the restoring beam."""

    n_iter: int
    """Iterations actually run."""

    converged: bool
    """True if the threshold was reached before ``max_iter``."""

    peak_history: np.ndarray
    """Residual peak at each iteration. Should fall; a plateau means trouble."""

    @property
    def model_flux(self) -> float:
        """Total flux in the component model, Jy."""
        return float(self.components.sum())

    @property
    def residual_rms(self) -> float:
        """RMS of the residual image, Jy/beam."""
        return float(np.std(self.residual))


def hogbom_clean(
    dirty: np.ndarray,
    dirty_beam: np.ndarray,
    gain: float = 0.1,
    max_iter: int = 2000,
    threshold: float | None = None,
    threshold_sigma: float = 3.0,
) -> CleanResult:
    """Run Hogbom CLEAN on a dirty image.

    Parameters
    ----------
    dirty : dirty image, Jy/beam, centred.
    dirty_beam : dirty beam, unit peak, same shape, centred.
    gain : loop gain. Fraction of the peak subtracted per iteration. Small
        values (0.05-0.2) are slower but more stable; a gain near 1 will
        overshoot and produce negative artefacts around bright sources.
    max_iter : hard iteration cap.
    threshold : absolute stopping threshold in Jy/beam. If ``None``, it is set
        to ``threshold_sigma`` times a robust noise estimate of the dirty image.
    threshold_sigma : used only when ``threshold`` is ``None``.

    Returns
    -------
    CleanResult

    Notes
    -----
    Stopping matters more than it looks, and it biases your photometry.

    Each iteration removes ``gain`` times the current peak, so the loop halts
    with roughly ``threshold`` worth of flux still sitting under every source.
    That is an *absolute* deficit, so the *fractional* error is worse for
    fainter sources. Measured on the noiseless three-source test field in
    ``tests/``, with ``gain=0.1``:

    ======================  =========  =========  ==========
    ``threshold_sigma``     1.00 Jy    0.50 Jy    0.25 Jy
    ======================  =========  =========  ==========
    3.0 (the default)       -3.7 %     -7.3 %     -16.1 %
    1.0                     -1.3 %     -2.6 %     -5.2 %
    0.3                     -0.4 %     -0.7 %     -1.6 %
    0.05                    -0.1 %     -0.1 %     -0.3 %
    ======================  =========  =========  ==========

    This is real CLEAN behaviour, not a defect in this implementation, and it
    is why faint-source flux scales in published catalogues need care.

    Clean too deep on *real* data, though, and you begin fitting noise, which
    manufactures spurious faint sources. The default of a few times the robust
    noise is the usual compromise; drop it only when, as in a noiseless
    simulation, there is no noise to fit.
    """
    if dirty.shape != dirty_beam.shape:
        raise ValueError("dirty image and dirty beam must have the same shape")
    if not 0 < gain <= 1:
        raise ValueError("gain must be in (0, 1]")

    n_pix = dirty.shape[0]
    centre = n_pix // 2

    if threshold is None:
        # Median absolute deviation: robust to the bright sources we are about
        # to remove, unlike a plain standard deviation.
        mad = np.median(np.abs(dirty - np.median(dirty)))
        threshold = threshold_sigma * 1.4826 * mad
        if threshold <= 0:
            # Degenerate images (an empty field, or one so sparse that more than
            # half the pixels are identical) have zero MAD. Without a floor the
            # loop would run to max_iter subtracting nothing.
            threshold = 1e-9 * float(np.max(np.abs(dirty)))

    residual = dirty.astype(float).copy()
    components = np.zeros_like(residual)
    peak_history = []
    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        flat_index = int(np.argmax(np.abs(residual)))
        row, col = divmod(flat_index, n_pix)
        peak = residual[row, col]
        peak_history.append(abs(peak))

        # `<=` rather than `<` so a perfectly empty field (peak 0, threshold 0)
        # terminates on the first iteration instead of spinning to max_iter.
        if abs(peak) <= threshold:
            converged = True
            n_iter -= 1
            peak_history.pop()
            break

        delta = gain * peak
        components[row, col] += delta
        # Shift the beam so its peak sits on the component, then subtract.
        # np.roll wraps, which is correct: the convolution is circular.
        residual -= delta * np.roll(
            dirty_beam, (row - centre, col - centre), axis=(0, 1)
        )

    clean_beam, beam_params = fit_clean_beam(dirty_beam)
    restored = convolve_fft(components, clean_beam) + residual

    return CleanResult(
        components=components,
        residual=residual,
        restored=restored,
        clean_beam=clean_beam,
        beam_params=beam_params,
        n_iter=n_iter,
        converged=converged,
        peak_history=np.asarray(peak_history),
    )
