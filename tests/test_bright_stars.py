"""Far-field light of bright stars (halo + spike ridges beyond the ePSF stamp)
and its wiring into the forward model behind default-off flags."""

from dataclasses import replace

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.psf import PSF
from euclid_polish.psf.psf_set import PSFSet
from euclid_polish.sky.generation.sky_simulator import _deposit_star
from euclid_polish.sky.observation.bright_stars import (
    FarFieldBand,
    _ray_entry,
    draw_jitter,
    draw_off_field_stars,
    far_field_flux,
    join_radii,
    join_weight,
    load_far_field_params,
    load_off_field_prior,
    render_far_field,
    spike_ridges,
)
from euclid_polish.sky.observation.observation_simulator import (
    ObservationSimulator,
    ObservationSimulatorConfig,
)

HR = 0.05
SPIKE_DEG = 17.0      # first ridge of the synthetic kernel; six, 60° apart


def _spiky_kernel(side: int, *, first_deg: float = SPIKE_DEG) -> np.ndarray:
    """Core + r^-3 halo + six r^-2 diffraction ridges, sum 1."""
    c = (side - 1) / 2.0
    yy, xx = np.indices((side, side), dtype=np.float64)
    dy, dx = yy - c, xx - c
    r = np.hypot(dy, dx)
    k = np.exp(-0.5 * (r / 2.0) ** 2) + 2e-3 * (1.0 + (r / 4.0) ** 2) ** -1.5
    for j in range(6):
        a = np.radians(first_deg + 60.0 * j)
        along = dx * np.cos(a) + dy * np.sin(a)
        perp = dy * np.cos(a) - dx * np.sin(a)
        k += np.where(along > 0, 5e-3 * (1.0 + along / 4.0) ** -2.0, 0.0) * np.exp(-0.5 * (perp / 1.5) ** 2)
    return k / k.sum()


def _sets(vis_side: int = 511, nisp_side: int = 1023) -> dict[str, PSFSet]:
    return {b.name: PSFSet.from_psfs([PSF(data=_spiky_kernel(vis_side if b.name == "VIS" else nisp_side),
                                          pixel_scale=HR)])
            for b in Config.BANDS}


def _scene(n: int, mags: list[tuple[int, int, float]]):
    stars = np.zeros((n, n, 4), np.float32)
    for x, y, m in mags:
        _deposit_star(stars, x, y, m, band_magnitudes=dict.fromkeys(Config.LR_INPUT_BAND_NAMES, m))
    img = Image(data=np.zeros_like(stars), pixel_scale_arcsec=HR,
                band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True)
    return img, stars


def _quiet(**kw) -> ObservationSimulatorConfig:
    return ObservationSimulatorConfig(add_noise=False, add_artifacts=False, add_saturation=False,
                                      randomize_psf=False, **kw)


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

def test_committed_far_field_parameters_are_physical():
    """Every band: halo ~ r^-3 and diffraction ridges ~ r^-2, ridges far above
    the halo, NISP ridges wider than VIS (longer wavelength, coarser pixels)."""
    p = load_far_field_params()
    assert set(p.bands) == {b.name for b in Config.BANDS}
    assert p.n_spikes == 6
    for name, b in p.bands.items():
        assert 2.6 < b.halo_gamma < 3.4, name
        assert 1.7 < b.spike_gamma < 2.3, name
        r = np.array([10.0, 20.0, 40.0])
        assert np.all(b.ridge_at(r) > 5.0 * b.halo_at(r)), name
        assert np.all(np.diff(np.log(b.halo_at(np.geomspace(4.0, 300.0, 50)))) < 0), name
    assert p.bands["VIS"].spike_fwhm_arcsec < 0.4 < p.bands["H_E"].spike_fwhm_arcsec


def test_profile_tables_are_continued_by_their_power_law_tails():
    b = FarFieldBand(r_arcsec=(1.0, 10.0), halo=(1.0, 1e-3), ridge=(1.0, 1e-2),
                     halo_gamma=3.0, spike_gamma=2.0, spike_fwhm_arcsec=0.3)
    assert b.halo_at(np.array([100.0]))[0] == pytest.approx(1e-6)
    assert b.ridge_at(np.array([100.0]))[0] == pytest.approx(1e-4)
    assert b.halo_at(np.array([0.1]))[0] == pytest.approx(1e3)          # first-segment slope
    with pytest.raises(ValueError):
        FarFieldBand(r_arcsec=(2.0, 1.0), halo=(1.0, 1.0), ridge=(1.0, 1.0),
                     halo_gamma=3.0, spike_gamma=2.0, spike_fwhm_arcsec=0.3)


def test_join_radii_stay_inside_the_stamps_own_taper():
    assert join_radii(25.575, (4.5, 6.5), 0.82) == (4.5, 6.5)
    r_in, r_out = join_radii(5.0, (4.5, 6.5), 0.82)
    assert r_out == pytest.approx(4.1) and r_out - r_in == pytest.approx(2.0)
    with pytest.raises(ValueError):
        join_radii(2.0, (4.5, 6.5), 0.82)
    w = join_weight(np.array([0.0, 4.5, 5.5, 6.5, 9.0]), 4.5, 6.5)
    np.testing.assert_allclose(w, [1.0, 1.0, 0.5, 0.0, 0.0], atol=1e-12)


# --------------------------------------------------------------------------- #
# Spike ridges come from the scene's own kernel
# --------------------------------------------------------------------------- #

def _angle_error(found_rad: np.ndarray, expect_deg: np.ndarray) -> float:
    d = (np.degrees(found_rad)[:, None] - expect_deg[None, :] + 180.0) % 360.0 - 180.0
    return float(np.abs(d).min(axis=1).max())


@pytest.mark.parametrize("roll", [0.0, 23.0, 211.0])
def test_spike_angles_follow_the_kernel_and_its_roll(roll):
    psf = PSF(data=_spiky_kernel(511), pixel_scale=HR)
    if roll:
        psf = psf.rotated(roll, order=3)
    angles, rel = spike_ridges(psf.data, HR, 6)
    # PSF.rotated turns the stamp counter-clockwise on the sky (rows up), i.e.
    # clockwise in array (row-down) coordinates.
    expect = (SPIKE_DEG - roll + 60.0 * np.arange(6)) % 360.0
    assert _angle_error(angles, expect) < 1.0
    assert rel.mean() == pytest.approx(1.0)


def test_kernel_without_ridges_gets_even_equal_ones():
    psf = PSF(data=_spiky_kernel(511), pixel_scale=HR)
    yy, xx = np.indices(psf.data.shape) - 255.0
    gauss = np.exp(-0.5 * (np.hypot(yy, xx) / 3.0) ** 2)
    angles, rel = spike_ridges(gauss / gauss.sum(), HR, 6)
    np.testing.assert_allclose(np.degrees(angles), 60.0 * np.arange(6))
    np.testing.assert_allclose(rel, 1.0)


# --------------------------------------------------------------------------- #
# Rendering: profile shape and flux normalisation
# --------------------------------------------------------------------------- #

def test_rendered_halo_and_ridge_follow_the_measured_profile():
    p = load_far_field_params()
    band = p.bands["H_E"]
    n, c, ps = 801, 400.0, 0.1
    angles = np.radians(SPIKE_DEG + 60.0 * np.arange(6))
    out = render_far_field((n, n), np.array([c]), np.array([c]), np.array([1e9]), band, p,
                           pixel_scale_arcsec=ps, angles=angles, rel=np.ones(6), r_in=4.5, r_out=6.5)
    for r in (10.0, 20.0, 35.0):
        # Halo: halfway between two ridges; ridge: on the first ridge.
        a_halo = np.radians(SPIKE_DEG + 30.0)
        yh, xh = c + r / ps * np.sin(a_halo), c + r / ps * np.cos(a_halo)
        assert out[round(yh), round(xh)] / 1e9 == pytest.approx(band.halo_at(np.array([np.hypot(
            round(yh) - c, round(xh) - c) * ps]))[0], rel=1e-3)
        a = angles[0]
        yr, xr = c + r / ps * np.sin(a), c + r / ps * np.cos(a)
        on = out[int(np.floor(yr)) - 1:int(np.floor(yr)) + 3, int(np.floor(xr)) - 1:int(np.floor(xr)) + 3].max() / 1e9
        expect = band.ridge_at(np.array([r]))[0] + band.halo_at(np.array([r]))[0]
        assert on == pytest.approx(expect, rel=0.15)
    # Nothing inside the join's inner radius.
    yy, xx = np.indices(out.shape)
    assert out[np.hypot(yy - c, xx - c) * ps < 4.5].max() == 0.0


def test_rendered_flux_matches_the_flux_normalisation():
    """The far field's summed light equals far_field_flux (same truncation),
    which is what the star kernel gives up."""
    p = replace(load_far_field_params(), r_max_arcsec=40.0)
    band = p.bands["H_E"]
    n, c, ps = 841, 420.0, 0.1
    out = render_far_field((n, n), np.array([c]), np.array([c]), np.array([1.0]), band, p,
                           pixel_scale_arcsec=ps, angles=np.radians(SPIKE_DEG + 60.0 * np.arange(6)),
                           rel=np.ones(6), r_in=4.5, r_out=6.5)
    assert out.sum() == pytest.approx(far_field_flux(band, p, 4.5, 6.5), rel=0.02)
    # Untruncated, a percent-level share of the light: VIS ~2%, NISP ~3%.
    full = load_far_field_params()
    for name in ("VIS", "H_E"):
        assert 0.005 < far_field_flux(full.bands[name], full, 4.5, 6.5) < 0.06


def test_amplitude_jitter_is_mean_one_and_seeded():
    p = load_far_field_params()
    h1, s1 = draw_jitter(20000, p, np.random.default_rng(5))
    h2, s2 = draw_jitter(20000, p, np.random.default_rng(5))
    np.testing.assert_array_equal(h1, h2)
    np.testing.assert_array_equal(s1, s2)
    assert h1.mean() == pytest.approx(1.0, abs=0.01) and s1.mean() == pytest.approx(1.0, abs=0.01)
    assert np.std(np.log(s1)) == pytest.approx(p.spike_jitter_ln, rel=0.05)


# --------------------------------------------------------------------------- #
# Forward model
# --------------------------------------------------------------------------- #

def test_far_field_extends_spikes_beyond_the_stamp_and_keeps_the_flux():
    n = 1200                                    # 60″ field, star in the middle
    img, stars = _scene(n, [(n // 2, n // 2, 10.0)])
    F = stars[n // 2, n // 2].astype(np.float64)
    off, _ = ObservationSimulator(psf_sets_by_band=_sets(), config=_quiet()).process(
        img, np.random.default_rng(0), star_hr_4ch=stars)
    on, _ = ObservationSimulator(psf_sets_by_band=_sets(),
                                 config=_quiet(bright_star_far_field=True, nisp_psf_taper=True)).process(
        img, np.random.default_rng(0), star_hr_4ch=stars)
    c = (n // 2 + 0.5) / 2 - 0.5
    k = Config.LR_INPUT_BAND_NAMES.index("VIS")
    a = np.radians(SPIKE_DEG)
    y, x = (round(c + 200 * np.sin(a)), round(c + 200 * np.cos(a)))   # 20″ out on a ridge
    assert off.data[y, x, k] == 0.0                # the 511² stamp ends at 12.8″
    assert on.data[y, x, k] > 1e-8 * F[k]
    for j in range(4):
        total = on.data[..., j].sum() / F[j]
        # The far field beyond a 30″ half-field holds well under 1% of the light.
        assert 0.99 < total < 1.0 + 1e-4, Config.LR_INPUT_BAND_NAMES[j]
        assert off.data[..., j].sum() / F[j] == pytest.approx(1.0, abs=1e-4)


def test_far_field_ridges_follow_the_scene_kernel_roll():
    n = 1000
    img, stars = _scene(n, [(n // 2, n // 2, 9.0)])
    rolled = {b: PSFSet.from_psfs([PSF(data=_spiky_kernel(511 if b == "VIS" else 1023, first_deg=47.0),
                                       pixel_scale=HR)]) for b in (bb.name for bb in Config.BANDS)}
    on, _ = ObservationSimulator(psf_sets_by_band=rolled, config=_quiet(bright_star_far_field=True)).process(
        img, np.random.default_rng(0), star_hr_4ch=stars)
    c = (n // 2 + 0.5) / 2 - 0.5
    k = Config.LR_INPUT_BAND_NAMES.index("VIS")
    th = np.radians(np.arange(0.0, 360.0, 0.25))
    prof = on.data[np.round(c + 180 * np.sin(th)).astype(int), np.round(c + 180 * np.cos(th)).astype(int), k]
    peak = np.degrees(th[np.argmax(prof)]) % 60.0
    assert abs(peak - 47.0 % 60.0) < 1.5


def test_far_field_is_seeded():
    n = 600
    img, stars = _scene(n, [(150, 300, 9.0), (450, 200, 12.0)])
    cfg = ObservationSimulatorConfig(bright_star_far_field=True, randomize_psf=False)
    sim = ObservationSimulator(psf_sets_by_band=_sets(), config=cfg)
    a, _ = sim.process(img, np.random.default_rng(11), star_hr_4ch=stars)
    b, _ = sim.process(img, np.random.default_rng(11), star_hr_4ch=stars)
    c, _ = sim.process(img, np.random.default_rng(12), star_hr_4ch=stars)
    np.testing.assert_array_equal(a.data, b.data)
    assert not np.array_equal(a.data, c.data)


def test_flags_off_leaves_kernels_and_output_untouched():
    """Off is the old forward model: kernels are the very objects passed in,
    nothing is priced, and the join knob is inert. (Bit-identity with the
    pre-change code was also checked on 12 reference scenes, 2026-11-27.)"""
    sets = _sets()
    n = 400
    img, stars = _scene(n, [(200, 200, 9.0)])
    base = ObservationSimulator(psf_sets_by_band=sets, config=ObservationSimulatorConfig())
    other = ObservationSimulator(psf_sets_by_band=sets,
                                 config=ObservationSimulatorConfig(bright_star_join_arcsec=(1.0, 2.0)))
    for name in sets:
        assert base._psf_sets[name] is sets[name]
    assert base._far_flux == {} and base._far_params is None
    a, _ = base.process(img, np.random.default_rng(3), star_hr_4ch=stars)
    b, _ = other.process(img, np.random.default_rng(3), star_hr_4ch=stars)
    np.testing.assert_array_equal(a.data, b.data)


def test_nisp_taper_alone_cuts_only_the_nisp_kernels():
    sets = _sets()
    sim = ObservationSimulator(psf_sets_by_band=sets, config=_quiet(nisp_psf_taper=True))
    assert sim._psf_sets["VIS"] is sets["VIS"]
    assert sim._far_flux == {}
    for name in ("Y_E", "J_E", "H_E"):
        k = sim._psf_sets[name].psfs[0].data
        assert k.shape[0] < 1023 and k.sum() == pytest.approx(1.0, abs=1e-6)
        c = (k.shape[0] - 1) / 2
        yy, xx = np.indices(k.shape)
        assert k[np.hypot(yy - c, xx - c) * HR > 6.5 + 1e-9].max() == 0.0


def test_config_rejects_bad_join():
    with pytest.raises(ValueError):
        ObservationSimulatorConfig(bright_star_join_arcsec=(6.5, 4.5))
    with pytest.raises(ValueError):
        ObservationSimulatorConfig(bright_star_min_peak_e=-1.0)


# --------------------------------------------------------------------------- #
# Off-field stars
# --------------------------------------------------------------------------- #

def test_off_field_draw_matches_counts_and_stays_outside():
    prior = load_off_field_prior()
    assert prior.mag_lo.min() == pytest.approx(6.5) and prior.mag_hi.max() == pytest.approx(16.0)
    assert np.all(prior.density_deg2 >= 0) and prior.colour_ratios.shape[1] == 3
    reach = np.full(prior.mag_lo.size, 60.0)
    shape, ps = (510, 510), HR
    hh = 0.5 * 510 * ps
    ring = (4 * (hh + 60.0) ** 2 - 4 * hh * hh) / 3600.0 ** 2
    expect = float(np.sum(prior.density_deg2) * ring)
    counts, rng = [], np.random.default_rng(0)
    for _ in range(400):
        ys, xs, mags, colour = draw_off_field_stars(rng, prior, reach, shape, ps)
        counts.append(ys.size)
        dy, dx = (ys - 254.5) * ps, (xs - 254.5) * ps
        assert np.all((np.abs(dy) >= hh) | (np.abs(dx) >= hh))
        assert np.all((np.abs(dy) <= hh + 60.0) & (np.abs(dx) <= hh + 60.0))
        assert np.all((mags >= 6.5) & (mags <= 16.0))
        assert np.all((colour >= 0) & (colour < len(prior.colour_ratios)))
    assert np.mean(counts) == pytest.approx(expect, abs=4 * np.sqrt(expect / 400))
    a = draw_off_field_stars(np.random.default_rng(9), prior, reach, shape, ps)
    b = draw_off_field_stars(np.random.default_rng(9), prior, reach, shape, ps)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_ray_entry():
    assert _ray_entry(-5.0, 2.0, 1.0, 0.0, 10.0, 10.0, 0.0) == pytest.approx(5.0)
    assert _ray_entry(-5.0, 2.0, -1.0, 0.0, 10.0, 10.0, 0.0) is None
    assert _ray_entry(-5.0, 12.0, 1.0, 0.0, 10.0, 10.0, 0.0) is None
    assert _ray_entry(-5.0, 12.0, 1.0, 0.0, 10.0, 10.0, 2.5) == pytest.approx(2.5)
    assert _ray_entry(5.0, 5.0, 0.6, 0.8, 10.0, 10.0, 0.0) == 0.0


def test_off_field_crossing_rate_is_dr1_like():
    """Fraction of 25.5″ fields crossed by off-field light ≥ 1σ in VIS. The
    DR1 counts and profiles give 12.4% (2000 simulated fields, 2026-11-27);
    the first report's spike-only estimate was 9%."""
    sim = ObservationSimulator(psf_sets_by_band=_sets(), config=_quiet(
        bright_star_far_field=True, nisp_psf_taper=True, off_field_stars=True))
    p, k = sim._far_params, Config.LR_INPUT_BAND_NAMES.index("VIS")
    angles, rel = spike_ridges(sim._psf_sets["VIS"].psfs[0].data, HR, 6)
    rng, hit = np.random.default_rng(1), []
    for _ in range(800):
        d = sim._draw_off_field((510, 510), rng)
        if d is None:
            hit.append(False)
            continue
        ys, xs, flux, hj, sj = d
        img = render_far_field((255, 255), (ys + 0.5) / 2 - 0.5, (xs + 0.5) / 2 - 0.5, flux[:, k],
                               p.bands["VIS"], p, pixel_scale_arcsec=0.1, angles=angles, rel=rel,
                               r_in=4.5, r_out=6.5, halo_jitter=hj, spike_jitter=sj,
                               min_peak_e=sim._off_floor_e["VIS"])
        hit.append(img.max() >= 21.0)
    assert 0.085 < np.mean(hit) < 0.17


def test_off_field_stars_light_the_field_but_not_the_target():
    n = 510
    img, stars = _scene(n, [])
    sim = ObservationSimulator(psf_sets_by_band=_sets(), config=_quiet(
        bright_star_far_field=True, nisp_psf_taper=True, off_field_stars=True))
    lit = 0
    for seed in range(40):
        lr, hr = sim.process(img, np.random.default_rng(seed), star_hr_4ch=stars)
        assert not hr.data.any()
        assert np.all(lr.data >= 0.0)
        lit += bool((lr.data[..., 0] > 21.0).any())
    assert 1 <= lit <= 20
    with pytest.raises(ValueError):
        ObservationSimulatorConfig(off_field_stars=True)


def test_off_field_kernel_spill_lands_at_the_edge():
    """A bright star just outside the edge spills its kernel core into the
    field (its own pixel stays outside)."""
    sim = ObservationSimulator(psf_sets_by_band=_sets(), config=_quiet(bright_star_far_field=True))
    psf = sim._psf_sets["VIS"].psfs[0]
    out = sim._stamp_kernel_at((100, 100), np.array([50]), np.array([-20]), np.array([1e6]), psf)
    assert out[50, 0] > 0 and out.sum() < 1e6 * 0.1
    far = sim._stamp_kernel_at((100, 100), np.array([50]), np.array([-2000]), np.array([1e6]), psf)
    assert not far.any()
