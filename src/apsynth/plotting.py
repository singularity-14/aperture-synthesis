"""Figures.

Colour choices here follow the job the data does, not taste:

* **Sky brightness, dirty image, restored image** carry *magnitude* - one hue,
  light to dark. A single-hue blue ramp with monotonically decreasing lightness.
  Never a rainbow map: the perceptual jumps in ``jet`` invent structure that is
  not in the data, which in this field has cost people real results.
* **Dirty beam and residuals** carry *polarity* - they are genuinely signed,
  with negative sidelobes that matter. Those get a diverging blue-to-red ramp
  with a neutral grey midpoint pinned at zero, so the eye reads "nothing" at
  zero and sign at a glance.

Every intensity panel carries a colourbar with units, because an astronomical
image without a brightness scale is undecodable.
"""

from __future__ import annotations

import astropy.units as u
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import AsinhNorm, LinearSegmentedColormap, TwoSlopeNorm

__all__ = [
    "SEQUENTIAL",
    "DIVERGING",
    "SERIES",
    "apply_style",
    "plot_array_layout",
    "plot_uv_coverage",
    "plot_dirty_beam",
    "plot_clean_summary",
    "plot_flux_recovery",
    "plot_convergence",
    "plot_coverage_comparison",
]

# -- palette ---------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: Categorical slots, in fixed order. Validated all-pairs for CVD separation.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

#: One-hue blue ramp, light to dark, for magnitude.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "apsynth_seq",
    ["#eef4fd", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"],
)

#: Blue to red with a neutral grey midpoint, for signed data.
DIVERGING = LinearSegmentedColormap.from_list(
    "apsynth_div",
    ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a08f", "#e34948", "#8f1f1f"],
)


def apply_style() -> None:
    """Set recessive, consistent chart chrome for every figure."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            # matplotlib ignores `grid(zorder=...)`; this is what actually keeps
            # gridlines behind the marks instead of drawing across them.
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "font.size": 9,
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def _symmetric_norm(data: np.ndarray) -> TwoSlopeNorm:
    """Diverging norm pinned at zero, so grey always means zero."""
    limit = float(np.max(np.abs(data))) or 1.0
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def _stretch_norm(data: np.ndarray) -> AsinhNorm:
    """Asinh stretch: linear near zero, logarithmic for bright values.

    Radio fields span a huge brightness range - here 0.06 Jy to 1 Jy, and real
    fields are far worse. On a linear scale the faint sources are invisible and
    the image looks empty. Asinh is the standard answer: it compresses the
    bright end without the log's blow-up at zero, so faint structure and
    negative sidelobes both stay readable.
    """
    peak = float(np.max(np.abs(data))) or 1.0
    return AsinhNorm(linear_width=0.02 * peak, vmin=float(np.min(data)), vmax=peak)


def _image_panel(ax, data, extent, cmap, norm=None, title="", unit="Jy/beam"):
    im = ax.imshow(
        data, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest"
    )
    ax.set_title(title)
    ax.set_xlabel('l offset (")')
    ax.set_ylabel('m offset (")')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(unit, fontsize=8, color=INK_SECONDARY)
    cbar.ax.tick_params(labelsize=7, colors=INK_MUTED)
    cbar.outline.set_edgecolor(AXIS)
    return im


# -- figures ---------------------------------------------------------------


def plot_array_layout(enu: np.ndarray, title: str = "Array layout", path=None):
    """Antenna positions in the local horizon frame."""
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(
        enu[:, 0], enu[:, 1], s=34, c=SERIES[0], edgecolor=SURFACE, linewidth=0.8, zorder=3
    )
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"{title}\n{len(enu)} antennas, {len(enu) * (len(enu) - 1) // 2} baselines")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def plot_uv_coverage(u_coords, v_coords, title="uv-coverage", path=None):
    """Sampled Fourier components, in kilowavelengths.

    Both the samples and their conjugates are shown, because both are measured.
    """
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    uu = np.concatenate([u_coords, -u_coords]) / 1e3
    vv = np.concatenate([v_coords, -v_coords]) / 1e3
    ax.scatter(uu, vv, s=0.6, c=SERIES[0], alpha=0.55, linewidths=0)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel(r"u (k$\lambda$)")
    ax.set_ylabel(r"v (k$\lambda$)")
    ax.set_title(title)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def plot_coverage_comparison(cases: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], path=None):
    """Side-by-side uv-coverage and resulting dirty beam for several cases.

    ``cases`` is a list of ``(label, u, v, dirty_beam)``. This is the figure
    that shows *why* earth rotation synthesis matters: more coverage, cleaner
    beam, fewer sidelobes to deconvolve.
    """
    n = len(cases)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7.0))
    axes = np.atleast_2d(axes)
    for col, (label, u_coords, v_coords, beam) in enumerate(cases):
        ax = axes[0, col]
        uu = np.concatenate([u_coords, -u_coords]) / 1e3
        vv = np.concatenate([v_coords, -v_coords]) / 1e3
        ax.scatter(uu, vv, s=0.4, c=SERIES[0], alpha=0.5, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(label)
        ax.set_xlabel(r"u (k$\lambda$)")
        if col == 0:
            ax.set_ylabel(r"v (k$\lambda$)")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        ax = axes[1, col]
        half = beam.shape[0] // 4
        centre = beam.shape[0] // 2
        crop = beam[centre - half : centre + half, centre - half : centre + half]
        im = ax.imshow(crop, origin="lower", cmap=DIVERGING, norm=_symmetric_norm(crop))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("dirty beam", fontsize=9)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=7, colors=INK_MUTED)
        cbar.outline.set_edgecolor(AXIS)
    fig.suptitle("More uv-coverage means a cleaner point spread function", y=0.98)
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def plot_dirty_beam(beam: np.ndarray, cell, zoom: int = 4, path=None):
    """Dirty beam image plus a horizontal cut showing the sidelobe level."""
    n_pix = beam.shape[0]
    centre = n_pix // 2
    half = max(8, n_pix // (2 * zoom))
    crop = beam[centre - half : centre + half, centre - half : centre + half]
    cell_arcsec = u.Quantity(cell).to_value(u.arcsec)
    extent = (half * cell_arcsec, -half * cell_arcsec, -half * cell_arcsec, half * cell_arcsec)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    _image_panel(
        axes[0], crop, extent, DIVERGING, _symmetric_norm(crop),
        "Dirty beam (point spread function)", "relative",
    )

    cut = beam[centre, :]
    offsets = (np.arange(n_pix) - centre) * cell_arcsec
    ax = axes[1]
    ax.axhline(0.0, color=AXIS, linewidth=0.8)
    ax.plot(offsets, cut, color=SERIES[0])
    peak_sidelobe = float(np.max(np.abs(cut[np.abs(offsets) > 2 * cell_arcsec])))
    ax.axhline(peak_sidelobe, color=SERIES[1], linewidth=1.2, linestyle="--")
    ax.annotate(
        f"peak sidelobe {peak_sidelobe:.1%}",
        xy=(offsets[int(0.72 * n_pix)], peak_sidelobe),
        xytext=(0, 8),
        textcoords="offset points",
        color=SERIES[1],
        fontsize=8,
        fontweight="bold",
    )
    ax.set_xlim(offsets[centre - half * 3], offsets[centre + half * 3])
    ax.set_xlabel('offset along l (")')
    ax.set_ylabel("relative response")
    ax.set_title("Horizontal cut through the beam")
    ax.grid(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def plot_clean_summary(sky, observation, result, path=None):
    """The five-panel story: truth, dirty, model, restored, residual.

    Two display conventions, both applied for legibility and both stated on the
    figure so nobody mistakes them for the data:

    * The two delta-function panels - the true sky and the CLEAN components -
      are **convolved with the restoring beam for display**. A point source
      occupies one pixel in 512x512 and is simply invisible otherwise. Showing
      them at the resolution the data actually support is also the honest
      comparison against the restored image.
    * All magnitude panels use an **asinh stretch**, so a 0.06 Jy source is
      visible beside a 1 Jy one.
    """
    cell_arcsec = u.Quantity(observation.cell).to_value(u.arcsec)
    n_pix = observation.n_pix
    half = n_pix / 2 * cell_arcsec
    extent = (half, -half, -half, half)

    # Imported here rather than at module scope to keep plotting a leaf module.
    from .clean import convolve_fft

    sky_display = convolve_fft(sky.image, result.clean_beam)
    model_display = convolve_fft(result.components, result.clean_beam)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))

    _image_panel(axes[0, 0], sky_display, extent, SEQUENTIAL,
                 _stretch_norm(sky_display),
                 "1. True sky (input model)", "Jy/beam")
    _image_panel(axes[0, 1], observation.dirty, extent, SEQUENTIAL,
                 _stretch_norm(observation.dirty),
                 "2. Dirty image = sky * dirty beam", "Jy/beam")
    _image_panel(axes[0, 2], model_display, extent, SEQUENTIAL,
                 _stretch_norm(model_display),
                 f"3. CLEAN components ({result.n_iter} iters)", "Jy/beam")
    _image_panel(axes[1, 0], result.restored, extent, SEQUENTIAL,
                 _stretch_norm(result.restored),
                 "4. Restored image", "Jy/beam")
    _image_panel(axes[1, 1], result.residual, extent, DIVERGING,
                 _symmetric_norm(result.residual), "5. Residual", "Jy/beam")

    ax = axes[1, 2]
    ax.axis("off")
    beam = result.beam_params
    summary = (
        f"Input total flux      {sky.total_flux:.3f} Jy\n"
        f"CLEAN model flux      {result.model_flux:.3f} Jy\n"
        f"Flux recovered        {100 * result.model_flux / sky.total_flux:.1f} %\n\n"
        f"Restoring beam        {beam['bmaj_pix'] * cell_arcsec:.2f}\" x "
        f"{beam['bmin_pix'] * cell_arcsec:.2f}\"\n"
        f"Beam position angle   {beam['bpa_deg']:.1f} deg\n\n"
        f"Residual RMS          {result.residual_rms:.2e} Jy/beam\n"
        f"Iterations            {result.n_iter}\n"
        f"Converged             {result.converged}\n"
        f"uv fill fraction      {observation.uv_fill_fraction:.3%}"
    )
    ax.text(0.0, 0.95, "Summary", fontsize=11, fontweight="bold",
            color=INK, va="top", transform=ax.transAxes)
    ax.text(0.0, 0.84, summary, fontsize=8.5, family="monospace",
            color=INK_SECONDARY, va="top", transform=ax.transAxes)
    ax.text(
        0.0, 0.18,
        "Display only: panels 1 and 3 are delta functions\n"
        "convolved with the restoring beam so they are\n"
        "visible; all magnitude panels use an asinh stretch.",
        fontsize=7.5, color=INK_MUTED, va="top", transform=ax.transAxes,
    )

    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def plot_flux_recovery(comparisons, path=None):
    """Input vs recovered flux, with the 1:1 line. The credibility figure."""
    inputs = np.array([c.input_flux for c in comparisons])
    recovered = np.array([c.recovered_flux for c in comparisons])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))

    ax = axes[0]
    lim = [0, float(max(inputs.max(), recovered.max())) * 1.12]
    ax.plot(lim, lim, color=AXIS, linewidth=1.2, linestyle="--", zorder=1)
    ax.annotate("1:1", xy=(lim[1] * 0.86, lim[1] * 0.90), color=INK_MUTED, fontsize=8)
    ax.scatter(inputs, recovered, s=58, c=SERIES[0], edgecolor=SURFACE,
               linewidth=1.0, zorder=3)
    for comparison in comparisons:
        ax.annotate(
            comparison.name,
            xy=(comparison.input_flux, comparison.recovered_flux),
            xytext=(6, -3), textcoords="offset points",
            fontsize=8, color=INK_SECONDARY,
        )
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Input flux (Jy)")
    ax.set_ylabel("Recovered flux (Jy)")
    ax.set_title("CLEAN flux recovery")
    ax.grid(True)

    ax = axes[1]
    errors = np.array([c.percent_error for c in comparisons])
    names = [c.name for c in comparisons]
    pad = max(1.0, float(np.abs(errors).max()) * 1.35)

    for tolerance in (-5.0, 5.0):
        ax.axvline(tolerance, color=AXIS, linewidth=1.0, linestyle="--", zorder=1)
    ax.annotate("+/-5%", xy=(5.0, len(names) - 0.6), xytext=(4, 0),
                textcoords="offset points", fontsize=7.5, color=INK_MUTED)
    ax.axvline(0.0, color=AXIS, linewidth=1.2, zorder=1)
    ax.barh(names, errors, color=SERIES[0], height=0.55, zorder=2)
    for name, error in zip(names, errors):
        ax.annotate(
            f"{error:+.2f}%",
            xy=(error, name),
            xytext=(6 if error >= 0 else -6, 0),
            textcoords="offset points",
            va="center", ha="left" if error >= 0 else "right",
            fontsize=8, color=INK_SECONDARY,
        )
    ax.set_xlim(-pad, pad)
    ax.set_xlabel("Flux error (%)")
    ax.set_title("Per-source error (fainter sources lose more)")
    ax.grid(True, axis="x")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    for axis in axes:
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def plot_convergence(result, path=None):
    """Residual peak against iteration, log scale."""
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.semilogy(np.arange(1, len(result.peak_history) + 1), result.peak_history,
                color=SERIES[0])
    rms = result.residual_rms
    ax.axhline(rms, color=SERIES[1], linewidth=1.4, linestyle="--")
    ax.annotate(f"residual RMS {rms:.1e}", xy=(1, rms), xytext=(6, 6),
                textcoords="offset points", color=SERIES[1],
                fontsize=8, fontweight="bold")
    ax.set_xlabel("CLEAN iteration")
    ax.set_ylabel("Residual peak (Jy/beam)")
    ax.set_title("CLEAN convergence")
    ax.grid(True, which="both")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig
