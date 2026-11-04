"""Quantitative checks on a reconstruction.

Pretty pictures are not evidence. These are the numbers that decide whether a
deconvolution actually worked:

* **Flux recovery** - does CLEAN return the flux density that went in? If not,
  every physical conclusion drawn from the image is wrong.
* **Dynamic range** - peak over residual RMS. The standard way radio images are
  quoted, and the figure of merit sidelobes degrade.
* **Residual RMS** - how close to thermal noise did we get?

Units discipline: the component image is in Jy/pixel, so summing it gives Jy
directly. The restored image is in Jy/beam, so summing *that* and calling it a
flux is wrong by the beam area - a classic and avoidable error, which is why
:func:`beam_area_pixels` exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np

from .sky import PointSource

__all__ = [
    "robust_rms",
    "dynamic_range",
    "beam_area_pixels",
    "FluxComparison",
    "compare_fluxes",
    "format_flux_table",
]


def robust_rms(image: np.ndarray) -> float:
    """Noise estimate via the median absolute deviation.

    Robust against real sources in the image, unlike ``np.std``. The 1.4826
    factor converts MAD to the equivalent Gaussian sigma.
    """
    mad = np.median(np.abs(image - np.median(image)))
    return float(1.4826 * mad)


def dynamic_range(image: np.ndarray, residual: np.ndarray) -> float:
    """Peak brightness divided by the residual RMS.

    The conventional quality figure for a radio image. A dirty image of a
    bright source is sidelobe-limited and has poor dynamic range no matter how
    sensitive the telescope; deconvolution is what buys it back.
    """
    rms = robust_rms(residual)
    if rms <= 0:
        return float("inf")
    return float(np.max(np.abs(image)) / rms)


def beam_area_pixels(bmaj_pix: float, bmin_pix: float) -> float:
    """Area of a Gaussian beam in pixels, given FWHMs in pixels.

    ``area = pi * bmaj * bmin / (4 ln 2)``.

    This is the number you divide a Jy/beam sum by to get Jy.
    """
    return float(np.pi * bmaj_pix * bmin_pix / (4.0 * np.log(2.0)))


@dataclass
class FluxComparison:
    """Input vs recovered flux for one source."""

    name: str
    l_arcsec: float
    m_arcsec: float
    input_flux: float
    recovered_flux: float

    @property
    def error(self) -> float:
        return self.recovered_flux - self.input_flux

    @property
    def percent_error(self) -> float:
        if self.input_flux == 0:
            return float("nan")
        return 100.0 * self.error / self.input_flux


def compare_fluxes(
    components: np.ndarray,
    sources: list[PointSource],
    cell,
    aperture_pix: int = 3,
) -> list[FluxComparison]:
    """Measure recovered flux for each input point source.

    CLEAN components for a single point source do not all land in one pixel:
    noise, gridding and the half-pixel position rounding scatter them into the
    neighbourhood. So flux is summed over a small square aperture rather than
    read from the central pixel.

    Parameters
    ----------
    components : CLEAN component image, Jy/pixel.
    sources : the input point sources.
    cell : image cell size as a ``Quantity`` angle.
    aperture_pix : half-width of the summing box, pixels.

    Returns
    -------
    One :class:`FluxComparison` per source.
    """
    n_pix = components.shape[0]
    cell_rad = u.Quantity(cell).to_value(u.rad)
    cell_arcsec = u.Quantity(cell).to_value(u.arcsec)
    results = []

    for index, source in enumerate(sources):
        l_rad = u.Quantity(source.l).to_value(u.rad)
        m_rad = u.Quantity(source.m).to_value(u.rad)
        col = int(round(l_rad / cell_rad)) + n_pix // 2
        row = int(round(m_rad / cell_rad)) + n_pix // 2

        r0, r1 = max(0, row - aperture_pix), min(n_pix, row + aperture_pix + 1)
        c0, c1 = max(0, col - aperture_pix), min(n_pix, col + aperture_pix + 1)
        recovered = float(components[r0:r1, c0:c1].sum())

        results.append(
            FluxComparison(
                name=f"S{index + 1}",
                l_arcsec=(col - n_pix // 2) * cell_arcsec,
                m_arcsec=(row - n_pix // 2) * cell_arcsec,
                input_flux=source.flux,
                recovered_flux=recovered,
            )
        )
    return results


def format_flux_table(comparisons: list[FluxComparison]) -> str:
    """Render flux comparisons as a Markdown table."""
    lines = [
        "| Source | l (\")| m (\")| Input (Jy) | Recovered (Jy) | Error (%) |",
        "|---|---|---|---|---|---|",
    ]
    for comparison in comparisons:
        lines.append(
            f"| {comparison.name} "
            f"| {comparison.l_arcsec:+.1f} "
            f"| {comparison.m_arcsec:+.1f} "
            f"| {comparison.input_flux:.3f} "
            f"| {comparison.recovered_flux:.3f} "
            f"| {comparison.percent_error:+.2f} |"
        )
    total_in = sum(c.input_flux for c in comparisons)
    total_out = sum(c.recovered_flux for c in comparisons)
    total_err = 100.0 * (total_out - total_in) / total_in if total_in else float("nan")
    lines.append(
        f"| **Total** | | | **{total_in:.3f}** | **{total_out:.3f}** "
        f"| **{total_err:+.2f}** |"
    )
    return "\n".join(lines)
