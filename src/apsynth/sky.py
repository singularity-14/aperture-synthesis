"""Sky models.

Images here are in **Jy/pixel**. That matters: after convolution with a dirty
beam normalised to unit peak, the resulting image is in **Jy/beam**, which is
the unit real radio images come in. Keeping the distinction straight is the
difference between getting source fluxes right and getting them wrong by the
beam area.

Coordinates ``l`` and ``m`` are direction cosines measured from the field
centre, small enough here to be treated as angles. ``l`` increases East (array
columns), ``m`` increases North (array rows).
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np

__all__ = ["PointSource", "GaussianSource", "SkyModel", "pixel_coords"]


@dataclass(frozen=True)
class PointSource:
    """An unresolved source.

    l, m : offsets from field centre as ``Quantity`` angles.
    flux : total flux density in Jy.
    """

    l: u.Quantity
    m: u.Quantity
    flux: float


@dataclass(frozen=True)
class GaussianSource:
    """A circular Gaussian source, ``fwhm`` as a ``Quantity`` angle."""

    l: u.Quantity
    m: u.Quantity
    flux: float
    fwhm: u.Quantity


def pixel_coords(n_pix: int, cell) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``l`` and ``m`` coordinate of every pixel, in radians.

    The field centre sits at pixel index ``n_pix // 2`` on both axes, matching
    the FFT convention used throughout :mod:`apsynth.imaging`.
    """
    cell_rad = u.Quantity(cell).to_value(u.rad)
    axis = (np.arange(n_pix) - n_pix // 2) * cell_rad
    l_grid, m_grid = np.meshgrid(axis, axis, indexing="xy")
    return l_grid, m_grid


class SkyModel:
    """A discretised sky brightness image in Jy/pixel."""

    def __init__(self, n_pix: int, cell):
        if n_pix % 2 != 0:
            raise ValueError("n_pix must be even so the centre lands on a pixel")
        self.n_pix = int(n_pix)
        self.cell = u.Quantity(cell)
        self.cell_rad = float(self.cell.to_value(u.rad))
        self.image = np.zeros((n_pix, n_pix), dtype=float)

    # -- construction -----------------------------------------------------

    def _to_pixel(self, l, m) -> tuple[int, int]:
        """Nearest pixel (row, col) for a sky offset, as (m index, l index)."""
        l_rad = u.Quantity(l).to_value(u.rad)
        m_rad = u.Quantity(m).to_value(u.rad)
        col = int(round(l_rad / self.cell_rad)) + self.n_pix // 2
        row = int(round(m_rad / self.cell_rad)) + self.n_pix // 2
        if not (0 <= row < self.n_pix and 0 <= col < self.n_pix):
            raise ValueError(f"source at l={l}, m={m} falls outside the image")
        return row, col

    def add_point(self, source: PointSource) -> "SkyModel":
        """Add a point source. Its flux lands entirely in one pixel.

        Note this quietly rounds the position to the pixel grid, so a source is
        displaced by up to half a pixel. That is a real (small) error and it
        shows up in the flux-recovery test as a sub-percent effect.
        """
        row, col = self._to_pixel(source.l, source.m)
        self.image[row, col] += source.flux
        return self

    def add_gaussian(self, source: GaussianSource) -> "SkyModel":
        """Add a circular Gaussian source, normalised to the requested total flux."""
        l_grid, m_grid = pixel_coords(self.n_pix, self.cell)
        l0 = u.Quantity(source.l).to_value(u.rad)
        m0 = u.Quantity(source.m).to_value(u.rad)
        fwhm = u.Quantity(source.fwhm).to_value(u.rad)
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        blob = np.exp(
            -((l_grid - l0) ** 2 + (m_grid - m0) ** 2) / (2.0 * sigma**2)
        )
        total = blob.sum()
        if total <= 0:
            raise ValueError("Gaussian source has no flux on the grid; fwhm too small?")
        self.image += source.flux * blob / total
        return self

    # -- properties -------------------------------------------------------

    @property
    def total_flux(self) -> float:
        """Total flux in the model, Jy."""
        return float(self.image.sum())

    def extent_arcsec(self) -> tuple[float, float, float, float]:
        """``(left, right, bottom, top)`` in arcsec, for ``matplotlib.imshow``."""
        half = self.n_pix / 2 * u.Quantity(self.cell).to_value(u.arcsec)
        return (half, -half, -half, half)  # l increases to the left, as on sky
