# Ensemble power-spectrum re-plot (eval-page style) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle the ensemble page's power-spectrum figure to match the evaluation page's look — VIS band, two panels T(k) and r(k), each model's line faint + the ensemble mean bold, on the test set.

**Architecture:** All curve data already exists in `EnsembleSpectrumAccumulator.curves()` (`k, P_hr, P_sr, r, P_members, r_members`). Add one pure helper that derives the plot arrays (θ, per-member T, mean T, r) and rewrite `render_ensemble_power_spectrum` to draw a VIS 1×2 (T, r) figure in the eval-page idiom (θ = 1/(2k) log x-axis, LR-sampling line, VIS PSF-FWHM guide). The caller in `ensemble_viz.py` is unchanged.

**Tech Stack:** Python, numpy, matplotlib (Agg), pytest.

---

### Task 1: Pure plot-curve helper `ensemble_ps_plot_curves`

**Files:**
- Modify: `euclid_polish/eval/power_spectrum.py` (add function after `coherence_half_scale`, ~line 251)
- Test: `tests/test_power_spectrum.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_power_spectrum.py`:

```python
from euclid_polish.eval.power_spectrum import ensemble_ps_plot_curves


def test_ensemble_ps_plot_curves_derives_per_member_transfer():
    k = np.array([0.5, 1.0, 2.0])
    curves = {
        "k": k,
        "P_hr": np.array([4.0, 4.0, 4.0]),
        "P_sr": np.array([1.0, 1.0, 1.0]),
        "r": np.array([0.9, 0.8, 0.7]),
        "P_members": np.array([[4.0, 1.0, 0.25],
                               [16.0, 4.0, 1.0]]),
        "r_members": np.array([[0.90, 0.80, 0.70],
                               [0.85, 0.75, 0.65]]),
    }
    out = ensemble_ps_plot_curves(curves)
    assert np.allclose(out["theta"], 0.5 / k)
    assert np.allclose(out["T"], np.sqrt(np.array([0.25, 0.25, 0.25])))
    assert np.allclose(out["T_members"][0], np.sqrt(np.array([1.0, 0.25, 0.0625])))
    assert np.allclose(out["T_members"][1], np.sqrt(np.array([4.0, 1.0, 0.25])))
    assert np.allclose(out["r"], curves["r"])
    assert out["r_members"].shape == (2, 3)


def test_ensemble_ps_plot_curves_handles_no_members():
    k = np.array([0.5, 1.0])
    curves = {"k": k, "P_hr": np.array([1.0, 1.0]),
              "P_sr": np.array([1.0, 1.0]), "r": np.array([1.0, 1.0])}
    out = ensemble_ps_plot_curves(curves)
    assert out["T_members"].shape == (0, 2)
    assert out["r_members"].shape == (0, 2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_power_spectrum.py::test_ensemble_ps_plot_curves_derives_per_member_transfer tests/test_power_spectrum.py::test_ensemble_ps_plot_curves_handles_no_members -v`
Expected: FAIL with `ImportError: cannot import name 'ensemble_ps_plot_curves'`.

- [ ] **Step 3: Implement the helper**

Add to `euclid_polish/eval/power_spectrum.py` immediately after `coherence_half_scale` (before `render_ensemble_power_spectrum`):

```python
def ensemble_ps_plot_curves(curves: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Derive the VIS T(k)/r(k) plot arrays (per-member + ensemble mean).

    Pure transform of an :meth:`EnsembleSpectrumAccumulator.curves` dict. Returns
    ``{theta, T, T_members, r, r_members}`` where ``theta = 1/(2k)`` arcsec,
    ``T = sqrt(P_sr/P_hr)`` (mean), ``T_members[j] = sqrt(P_members[j]/P_hr)``,
    ``r`` is the ensemble mean vs HR, ``r_members`` the per-member correlations.
    Member arrays are ``(M, nbins)`` and empty ``(0, nbins)`` when <2 members.
    """
    k = np.asarray(curves.get("k", []), float)
    nan_k = np.full(k.size, np.nan)
    p_hr = np.asarray(curves.get("P_hr", nan_k), float)
    p_sr = np.asarray(curves.get("P_sr", nan_k), float)
    r = np.asarray(curves.get("r", nan_k), float)
    p_members = np.asarray(curves.get("P_members", np.empty((0, k.size))), float)
    r_members = np.asarray(curves.get("r_members", np.empty((0, k.size))), float)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = 0.5 / k
        t_mean = np.sqrt(p_sr / p_hr)
        t_members = (np.sqrt(p_members / p_hr[None, :])
                     if p_members.size else np.empty((0, k.size)))
    return {"theta": theta, "T": t_mean, "T_members": t_members,
            "r": r, "r_members": r_members}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_power_spectrum.py::test_ensemble_ps_plot_curves_derives_per_member_transfer tests/test_power_spectrum.py::test_ensemble_ps_plot_curves_handles_no_members -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/eval/power_spectrum.py tests/test_power_spectrum.py
git commit -m "ensemble PS: pure helper deriving per-member/mean T(k), r(k), theta"
```

---

### Task 2: Rewrite `render_ensemble_power_spectrum` to eval-page style (VIS, T + r)

**Files:**
- Modify: `euclid_polish/eval/power_spectrum.py:253-327` (replace the function body)
- Test: `tests/test_power_spectrum.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_power_spectrum.py`:

```python
def test_render_ensemble_power_spectrum_writes_png(tmp_path):
    from euclid_polish.eval.power_spectrum import render_ensemble_power_spectrum

    k = np.geomspace(0.2, 10.0, 24)
    rng = np.random.default_rng(0)
    curves = {
        "k": k,
        "P_hr": np.abs(rng.normal(1e3, 10, k.size)),
        "P_sr": np.abs(rng.normal(5e2, 10, k.size)),
        "P_disagree": np.abs(rng.normal(10, 1, k.size)),
        "r": np.clip(1.0 - k / 20.0, 0, 1),
        "T": np.clip(1.0 - k / 40.0, 0, 1),
        "P_members": np.abs(rng.normal(5e2, 20, (5, k.size))),
        "r_members": np.clip(
            1.0 - k[None, :] / 20.0 + rng.normal(0, 0.02, (5, k.size)), 0, 1),
    }
    out = tmp_path / "ps.png"
    res = render_ensemble_power_spectrum(str(out), curves, n_fields=12)
    assert res == str(out)
    assert out.is_file() and out.stat().st_size > 0


def test_render_ensemble_power_spectrum_empty_returns_none(tmp_path):
    from euclid_polish.eval.power_spectrum import render_ensemble_power_spectrum

    out = tmp_path / "ps.png"
    assert render_ensemble_power_spectrum(
        str(out), {"k": np.array([]), "P_hr": np.array([])}) is None
    assert not out.exists()
```

- [ ] **Step 2: Run the test to verify current behaviour**

Run: `pytest tests/test_power_spectrum.py::test_render_ensemble_power_spectrum_writes_png -v`
Expected: PASS already (the old two-panel renderer also writes a PNG) — this test guards the contract; the visual change is verified in Step 4/manual. `test_render_ensemble_power_spectrum_empty_returns_none` should also PASS.

- [ ] **Step 3: Replace the renderer**

Replace `euclid_polish/eval/power_spectrum.py:253-327` (the whole `render_ensemble_power_spectrum` function) with:

```python
def render_ensemble_power_spectrum(out_png: str, curves: dict[str, np.ndarray],
                                   *, n_fields: int = 0) -> str | None:
    """VIS ensemble power spectrum in the evaluation-page idiom (test fields).

    Two panels — left the transfer function ``T(k) = sqrt(P_SR/P_HR)``, right the
    cross-correlation ``r(k)`` — each drawn against angular scale ``theta =
    1/(2k)`` (arcsec) on a log axis. Every ensemble member is a faint line; the
    ensemble mean is bold. Guides: LR sampling (0.10") and the VIS PSF FWHM.
    Returns the path, or ``None`` when there is no finite HR power.
    """
    k = np.asarray(curves.get("k", []), float)
    if k.size == 0 or not np.isfinite(np.asarray(curves.get("P_hr", []), float)).any():
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    cv = ensemble_ps_plot_curves(curves)
    x = cv["theta"]
    vis_color = BAND_COLORS["VIS"]
    vis_fwhm = float(Config.get_band("VIS").psf_fwhm_arcsec)
    lr_scale = 0.5 / LR_NYQUIST_CYC_ARCSEC                 # 0.10" LR sampling
    scale_lo = float(Config.DEFAULT_PIXEL_SCALE)           # 0.05" HR pixel
    scale_hi = float(np.nanmax(x)) if np.isfinite(x).any() else 1.0
    xticks = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    panels = (
        (ax_t, cv["T"], cv["T_members"],
         "transfer function  T = √(P_SR/P_HR)", (0.0, 1.45), "T(k)  [VIS]"),
        (ax_r, cv["r"], cv["r_members"],
         "cross-correlation  r = P_HR×SR/√(P_HR·P_SR)", (0.0, 1.05), "r(k)  [VIS]"),
    )
    for ax, mean_curve, member_curves, title, ylim, ylabel in panels:
        for j, row in enumerate(member_curves):
            ax.plot(x, row, color=vis_color, lw=0.7, alpha=0.30,
                    label=("individual models" if j == 0 else None))
        ax.plot(x, mean_curve, "-o", ms=3.0, lw=2.0, color=vis_color,
                label="ensemble mean")
        ax.axhline(1.0, ls=":", color="#888", lw=1.0)
        ax.axvline(lr_scale, ls="--", color="#333", lw=1.3,
                   label="LR sampling (0.1″)")
        ax.axvline(vis_fwhm, ls=(0, (5, 2)), lw=1.5, alpha=0.55, color=vis_color,
                   label="VIS PSF FWHM")
        ax.set_xscale("log")
        ax.set_xlim(scale_lo, scale_hi)
        ax.set_ylim(*ylim)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("angular scale  θ = 1/2k  [arcsec]   (0.05″ = HR pixel)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.set_title(title)
    ax_r.legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Ensemble angular power spectrum — VIS (test fields"
        + (f", {n_fields} fields" if n_fields else "") + ")\n"
        "each line = one model · bold = ensemble mean · finer (smaller θ) → left",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png
```

Note: `coherence_half_scale` remains in the module (still used elsewhere/tests); it's simply no longer called here. `rho`/`P_disagree` stay in `curves()` for the JSON sidecar.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_power_spectrum.py -v`
Expected: PASS (all, including the two new render tests and the math tests).

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/eval/power_spectrum.py tests/test_power_spectrum.py
git commit -m "ensemble PS: eval-page style VIS T(k)+r(k), per-member + mean lines"
```

---

### Task 3: Manual verification + full suite

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -q`
Expected: PASS (no regressions).

- [ ] **Step 2: Regenerate and eyeball the plot**

The ensemble eval job (`euclid_polish/web/helpers/ensemble_viz.py`) calls
`render_ensemble_power_spectrum(ps_png, curves, n_fields=...)` unchanged. If a prior
`data/vis/ensemble/ensemble_power_spectrum.json` exists, render from it:

Run:
```bash
python -c "import json,numpy as np; from euclid_polish.eval.power_spectrum import render_ensemble_power_spectrum as R; c={k:(np.array(v) if isinstance(v,list) else v) for k,v in json.load(open('data/vis/ensemble/ensemble_power_spectrum.json')).items()}; print(R('data/vis/ensemble/ensemble_power_spectrum.png', c, n_fields=int(c['k'].size>0)))"
```
Expected: prints the PNG path; open it and confirm two VIS panels (T left, r right), faint per-member lines, bold mean, θ x-axis, LR + PSF guides. (If no JSON exists yet, skip — this is covered by the unit test.)

- [ ] **Step 3: Commit any doc/tweak if needed** (else nothing to do).

---

## Self-review notes

- Spec §1 coverage: VIS band ✅, T(k)+r(k) panels ✅, per-member faint + mean bold ✅, eval-page θ axis + guides ✅, test-set source unchanged (caller untouched) ✅, ρ off by default ✅ (kept in `curves()`, not plotted).
- No placeholders; every code step is complete.
- Type consistency: helper returns `theta/T/T_members/r/r_members`; renderer consumes exactly those keys.
