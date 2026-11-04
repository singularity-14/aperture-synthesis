"""End-to-end demonstration: array -> uv-coverage -> dirty image -> CLEAN.

Writes every figure in ``figures/`` and the numerical results in ``results/``.

Run with::

    python scripts/run_demo.py

The demo deliberately runs the same field twice:

1. **Noiseless, cleaned deep.** Asks whether the algorithm itself conserves
   flux, with nothing else in the way.
2. **With thermal noise, cleaned to 3 sigma.** What you would actually do with
   real data, and what it costs you in photometric accuracy.

Reporting only the first would be flattering and misleading; reporting only
the second would hide whether the implementation is correct.
"""

from __future__ import annotations

import json
from pathlib import Path

import astropy.units as u
import numpy as np

from apsynth import arrays, clean, imaging, metrics, plotting, sky, uvcoverage

# -- configuration ---------------------------------------------------------

N_PIX = 512
CELL = 1.5 * u.arcsec
FREQUENCY = 1.4 * u.GHz
LATITUDE = 34 * u.deg
DECLINATION = 45 * u.deg
N_PER_ARM = 9
R_MAX = 2500.0  # metres
HOUR_ANGLES = np.linspace(-4.0, 4.0, 240) * u.hourangle
NOISE_JY = 0.35
SEED = 20261102

SOURCES = [
    sky.PointSource(l=0 * u.arcsec, m=0 * u.arcsec, flux=1.000),
    sky.PointSource(l=90 * u.arcsec, m=60 * u.arcsec, flux=0.500),
    sky.PointSource(l=-120 * u.arcsec, m=45 * u.arcsec, flux=0.250),
    sky.PointSource(l=60 * u.arcsec, m=-110 * u.arcsec, flux=0.125),
    sky.PointSource(l=-75 * u.arcsec, m=-80 * u.arcsec, flux=0.060),
]

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"


def build_coverage(hour_angles):
    """Antenna layout -> uv samples in wavelengths."""
    enu = arrays.y_array(n_per_arm=N_PER_ARM, r_max=R_MAX)
    xyz = uvcoverage.enu_to_xyz(enu, LATITUDE)
    baselines = uvcoverage.baseline_vectors(xyz)
    tracks = uvcoverage.uvw_tracks(
        baselines, hour_angles, DECLINATION, uvcoverage.wavelength(FREQUENCY)
    )
    return enu, baselines, uvcoverage.uv_points(tracks)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    plotting.apply_style()
    rng = np.random.default_rng(SEED)

    print("=" * 68)
    print("Aperture synthesis demo")
    print("=" * 68)

    # -- 1. array and coverage --------------------------------------------
    enu, baselines, (u_coords, v_coords) = build_coverage(HOUR_ANGLES)
    lam = uvcoverage.wavelength(FREQUENCY)
    b_max_m = float(np.max(np.linalg.norm(baselines, axis=1)))
    resolution = imaging.max_baseline_resolution(u_coords, v_coords)

    print(f"\nAntennas            {len(enu)}")
    print(f"Baselines           {len(baselines)}")
    print(f"Integrations        {len(HOUR_ANGLES)}")
    print(f"uv samples          {u_coords.size:,} (x2 with conjugates)")
    print(f"Observing frequency {FREQUENCY}  (lambda = {lam * 100:.2f} cm)")
    print(f"Longest baseline    {b_max_m:.0f} m = {b_max_m / lam / 1e3:.1f} klambda")
    print(f"Expected resolution {resolution:.2f}  (~{resolution / CELL:.1f} pixels)")

    plotting.plot_array_layout(
        enu, "VLA-like Y configuration", FIGURES / "01_array_layout.png"
    )
    plotting.plot_uv_coverage(
        u_coords,
        v_coords,
        f"uv-coverage: {len(HOUR_ANGLES)} integrations over 8 hours",
        FIGURES / "02_uv_coverage.png",
    )

    # -- 2. why earth rotation synthesis matters ---------------------------
    cases = []
    for label, has in [
        ("snapshot (1 integration)", np.array([0.0]) * u.hourangle),
        ("1 hour", np.linspace(-0.5, 0.5, 30) * u.hourangle),
        ("8 hours", HOUR_ANGLES),
    ]:
        _, _, (uu, vv) = build_coverage(has)
        sampling_case = imaging.grid_samples(uu, vv, N_PIX, CELL)
        obs_case = imaging.invert(np.zeros((N_PIX, N_PIX)), sampling_case, cell=CELL)
        cases.append((label, uu, vv, obs_case.beam))
    plotting.plot_coverage_comparison(cases, FIGURES / "03_coverage_comparison.png")

    # -- 3. the sky model --------------------------------------------------
    model = sky.SkyModel(n_pix=N_PIX, cell=CELL)
    for source in SOURCES:
        model.add_point(source)
    print(f"\nSky model           {len(SOURCES)} point sources, "
          f"{model.total_flux:.3f} Jy total")

    sampling = imaging.grid_samples(u_coords, v_coords, N_PIX, CELL)

    # -- 4. noiseless run: is the algorithm flux-conserving? ---------------
    print("\n" + "-" * 68)
    print("Run A - noiseless, cleaned deep (threshold_sigma=0.05)")
    print("-" * 68)
    obs_clean_data = imaging.invert(model.image, sampling, cell=CELL)
    result_deep = clean.hogbom_clean(
        obs_clean_data.dirty, obs_clean_data.beam,
        gain=0.1, max_iter=50000, threshold_sigma=0.05,
    )
    fluxes_deep = metrics.compare_fluxes(result_deep.components, SOURCES, CELL)
    print(metrics.format_flux_table(fluxes_deep))
    print(f"\nIterations {result_deep.n_iter}, converged {result_deep.converged}")

    plotting.plot_dirty_beam(obs_clean_data.beam, CELL, path=FIGURES / "04_dirty_beam.png")

    # -- 5. realistic run: with noise, cleaned to 3 sigma ------------------
    print("\n" + "-" * 68)
    print(f"Run B - thermal noise, cleaned to 3 sigma (what you'd really do)")
    print("-" * 68)
    observation = imaging.invert(
        model.image, sampling, cell=CELL, noise_jy=NOISE_JY, rng=rng
    )
    result = clean.hogbom_clean(
        observation.dirty, observation.beam, gain=0.1, max_iter=50000, threshold_sigma=3.0
    )
    fluxes = metrics.compare_fluxes(result.components, SOURCES, CELL)
    print(metrics.format_flux_table(fluxes))

    dr_before = metrics.dynamic_range(observation.dirty, observation.dirty)
    dr_after = metrics.dynamic_range(result.restored, result.residual)
    beam = result.beam_params
    cell_arcsec = CELL.to_value(u.arcsec)

    print(f"\nRestoring beam      {beam['bmaj_pix'] * cell_arcsec:.2f}\" x "
          f"{beam['bmin_pix'] * cell_arcsec:.2f}\" at {beam['bpa_deg']:.1f} deg")
    print(f"Residual RMS        {result.residual_rms:.3e} Jy/beam")
    print(f"Dynamic range       {dr_before:.1f} (dirty) -> {dr_after:.1f} (restored)")
    print(f"Iterations          {result.n_iter}, converged {result.converged}")
    print(f"uv fill fraction    {observation.uv_fill_fraction:.3%}")

    plotting.plot_clean_summary(model, observation, result, FIGURES / "05_clean_summary.png")
    plotting.plot_flux_recovery(fluxes, FIGURES / "06_flux_recovery.png")
    plotting.plot_convergence(result, FIGURES / "07_convergence.png")

    # -- 6. where CLEAN struggles ------------------------------------------
    print("\n" + "-" * 68)
    print("Run C - extended emission, where CLEAN's point-source prior fails")
    print("-" * 68)
    extended = sky.SkyModel(n_pix=N_PIX, cell=CELL)
    extended.add_gaussian(
        sky.GaussianSource(0 * u.arcsec, 0 * u.arcsec, flux=1.0, fwhm=70 * u.arcsec)
    )
    extended.add_point(sky.PointSource(120 * u.arcsec, 90 * u.arcsec, flux=0.4))
    obs_ext = imaging.invert(extended.image, sampling, cell=CELL, noise_jy=NOISE_JY, rng=rng)
    result_ext = clean.hogbom_clean(
        obs_ext.dirty, obs_ext.beam, gain=0.1, max_iter=50000, threshold_sigma=3.0
    )
    extended_recovery = 100 * result_ext.model_flux / extended.total_flux
    print(f"Input flux          {extended.total_flux:.3f} Jy "
          f"(1.0 Jy extended + 0.4 Jy compact)")
    print(f"CLEAN model flux    {result_ext.model_flux:.3f} Jy")
    print(f"Recovered           {extended_recovery:.1f} %")
    print("The smooth component is poorly represented by delta functions, and")
    print("short spacings missing from the uv-coverage resolve it out.")
    plotting.plot_clean_summary(
        extended, obs_ext, result_ext, FIGURES / "08_extended_source.png"
    )

    # -- 7. persist results -------------------------------------------------
    summary = {
        "configuration": {
            "n_antennas": len(enu),
            "n_baselines": len(baselines),
            "n_integrations": len(HOUR_ANGLES),
            "frequency_ghz": FREQUENCY.to_value(u.GHz),
            "latitude_deg": LATITUDE.to_value(u.deg),
            "declination_deg": DECLINATION.to_value(u.deg),
            "n_pix": N_PIX,
            "cell_arcsec": cell_arcsec,
            "max_baseline_m": b_max_m,
            "noise_jy_per_visibility": NOISE_JY,
            "seed": SEED,
        },
        "beam": {
            "expected_resolution_arcsec": float(resolution.to_value(u.arcsec)),
            "fitted_bmaj_arcsec": beam["bmaj_pix"] * cell_arcsec,
            "fitted_bmin_arcsec": beam["bmin_pix"] * cell_arcsec,
            "fitted_bpa_deg": beam["bpa_deg"],
            "uv_fill_fraction": observation.uv_fill_fraction,
        },
        "run_a_noiseless_deep": {
            "threshold_sigma": 0.05,
            "iterations": result_deep.n_iter,
            "model_flux_jy": result_deep.model_flux,
            "input_flux_jy": model.total_flux,
            "worst_percent_error": max(abs(f.percent_error) for f in fluxes_deep),
            "per_source": [
                {"name": f.name, "input_jy": f.input_flux,
                 "recovered_jy": f.recovered_flux, "percent_error": f.percent_error}
                for f in fluxes_deep
            ],
        },
        "run_b_noisy_3sigma": {
            "threshold_sigma": 3.0,
            "iterations": result.n_iter,
            "model_flux_jy": result.model_flux,
            "residual_rms_jy_per_beam": result.residual_rms,
            "dynamic_range_dirty": dr_before,
            "dynamic_range_restored": dr_after,
            "per_source": [
                {"name": f.name, "input_jy": f.input_flux,
                 "recovered_jy": f.recovered_flux, "percent_error": f.percent_error}
                for f in fluxes
            ],
        },
        "run_c_extended": {
            "input_flux_jy": extended.total_flux,
            "model_flux_jy": result_ext.model_flux,
            "percent_recovered": extended_recovery,
        },
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    table = "\n\n".join([
        "## Run A - noiseless, cleaned deep (threshold_sigma = 0.05)",
        "Does the algorithm conserve flux when nothing else is in the way?",
        metrics.format_flux_table(fluxes_deep),
        "## Run B - with thermal noise, cleaned to 3 sigma",
        "What you would actually do with real data, and what it costs.",
        metrics.format_flux_table(fluxes),
        "## Run C - extended emission",
        f"Input {extended.total_flux:.3f} Jy, CLEAN model {result_ext.model_flux:.3f} Jy "
        f"({extended_recovery:.1f} % recovered).",
    ])
    (RESULTS / "flux_table.md").write_text(table, encoding="utf-8")

    print(f"\nFigures written to  {FIGURES}")
    print(f"Results written to  {RESULTS}")


if __name__ == "__main__":
    main()
