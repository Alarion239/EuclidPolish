# Codebase Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the seven reproduced defects from the July 2026 review, preserve the zero-login localhost workflow, and add a truthful minimal CI quality gate.

**Architecture:** Harden boundaries centrally: one Flask unsafe-request guard, one loopback bind validator, one thread-local stream-routing layer, one evaluation run resolver, and lazy package export maps. Preserve existing public APIs and on-disk compatibility while adding narrow regression coverage for every changed behavior.

**Tech Stack:** Python 3.12, Flask, NumPy/Astropy, pytest, Ruff, GitHub Actions with Conda.

**Spec:** `docs/superpowers/specs/2026-07-12-codebase-hardening-design.md`

**Verification constraint:** Never run the full pytest suite. Run only the named files/nodes below, plus `pytest --collect-only`, `compileall`, and Ruff. Set `NUMBA_DISABLE_JIT=1` for pytest commands.

---

### Task 1: Protect localhost Web mutations

**Files:**
- Modify: `euclid_polish/web/app.py`
- Create: `tests/test_web_security.py`

- [ ] Add failing Flask-client tests proving a cross-origin `POST` returns 403 without invoking a patched mutation, a same-origin `POST` succeeds, a cross-site Fetch Metadata request is rejected, and a headerless local client remains compatible.
- [ ] Run `NUMBA_DISABLE_JIT=1 pytest tests/test_web_security.py -q`; confirm the hostile request currently reaches the mutation.
- [ ] Add a central `before_request` guard for `POST`, `PUT`, `PATCH`, and `DELETE`. Compare parsed `Origin` scheme/authority with the request effective origin; reject `Sec-Fetch-Site: cross-site`; return a concise 403 without route execution.
- [ ] Add a pure loopback-host validator and CLI tests for `127.0.0.1`, `::1`, and `localhost`; reject other bind hosts before `app.run()`.
- [ ] Rerun only `tests/test_web_security.py` until green.

### Task 2: Isolate concurrent job output and state

**Files:**
- Modify: `euclid_polish/web/jobs.py`
- Create: `tests/test_jobs.py`

- [ ] Add a coordinated two-thread regression test whose expected logs are `A1\nA2\n` and `B1\n`, plus a polling/serialization smoke test.
- [ ] Run `NUMBA_DISABLE_JIT=1 pytest tests/test_jobs.py -q`; confirm the existing process-global redirects cross-contaminate logs.
- [ ] Replace `redirect_stdout`/`redirect_stderr` with process-wide stream proxies that route writes through thread-local bindings and fall back to the original streams.
- [ ] Add a per-`Job` reentrant lock around log writes/reads, progress mutation, and dictionary serialization. Ensure stream bindings are cleared in `finally`.
- [ ] Rerun only `tests/test_jobs.py` until green.

### Task 3: Make catalog CSV persistence lossless

**Files:**
- Modify: `euclid_polish/catalog/catalog_object.py`
- Modify: the existing catalog-object test module discovered with `rg "CatalogObject" tests`

- [ ] Add failing row and CSV round-trip tests for `kind="galaxy"`, `ra=0.0`, and `dec=0.0`, plus a legacy row without `kind` that defaults to `star`.
- [ ] Run only the selected catalog test module; confirm kind and zero coordinates are lost.
- [ ] Add `kind` to the base columns and row serialization. Deserialize coordinates with explicit `None` checks and normalize missing/empty/non-string kinds to `star`.
- [ ] Rerun only the selected catalog test module until green.

### Task 4: Honor selected evaluation runs safely

**Files:**
- Modify: `euclid_polish/web/routes/evaluation.py`
- Modify: `tests/test_eval_catalog.py`

- [ ] Seed distinct root and child manifests in failing API tests; prove `?run=child` must return child data and unsafe/missing names are rejected.
- [ ] Run `NUMBA_DISABLE_JIT=1 pytest tests/test_eval_catalog.py -q -k 'runs_api or run_selection'`; confirm the endpoint ignores the valid child selection.
- [ ] Implement one resolver for omitted/root aliases and direct child run directories. Reject separators, dot segments, symlink escapes, missing directories, and missing required data.
- [ ] Replace every ad hoc `run` selection in evaluation routes with the resolver while preserving each route's existing response style.
- [ ] Rerun the focused `test_eval_catalog.py` selection until green.

### Task 5: Preserve PSF provenance on resampling

**Files:**
- Modify: `euclid_polish/psf/core.py`
- Modify: `tests/test_psf_class.py`

- [ ] Add a failing test asserting resampling changes data/scale but preserves the provenance stamp, while the no-op returns `self`.
- [ ] Run `NUMBA_DISABLE_JIT=1 pytest tests/test_psf_class.py -q -k resampl`; confirm the stamp disappears.
- [ ] Implement resampling with `dataclasses.replace(self, data=..., pixel_scale=...)`.
- [ ] Rerun the same focused nodes until green.

### Task 6: Remove eager sky-generation imports

**Files:**
- Modify: `euclid_polish/sky/__init__.py`
- Modify: `euclid_polish/sky/generation/__init__.py`
- Create: `tests/test_sky_lazy_imports.py`

- [ ] Add subprocess tests importing the observation-only path and asserting `lenstronomy` and lens-generation modules are absent from `sys.modules`; add compatibility checks for representative legacy re-exports.
- [ ] Run `NUMBA_DISABLE_JIT=1 pytest tests/test_sky_lazy_imports.py -q`; confirm eager imports fail the boundary assertion.
- [ ] Replace eager imports with explicit public-name-to-module maps, `__all__`, module `__getattr__`, and `__dir__`. Cache resolved names in module globals.
- [ ] Rerun only `tests/test_sky_lazy_imports.py` until green.

### Task 7: Repair the stale saturation test

**Files:**
- Modify: `tests/test_measure_star_saturation.py`

- [ ] Change the removed private-helper import to `euclid_polish.photometry.adu_per_s_to_electrons_factor` and keep the existing behavioral assertion.
- [ ] Run `NUMBA_DISABLE_JIT=1 pytest tests/test_measure_star_saturation.py -q`; confirm collection and execution succeed.

### Task 8: Establish a clean, focused CI gate

**Files:**
- Modify: `environment.yml`
- Modify: Ruff-reported files only
- Create: `.github/workflows/quality.yml`

- [ ] Add Ruff to the Conda environment.
- [ ] Run `ruff check . --fix` for safe mechanical fixes, inspect the diff, then resolve remaining findings individually without behavior changes.
- [ ] Run `ruff check .` and require zero findings.
- [ ] Add a GitHub Actions workflow using `actions/checkout@v4` and `conda-incubator/setup-miniconda@v4`; run compilation, Ruff, collection-only with JIT disabled, and exactly the focused regression files from Tasks 1–7. Do not add a full-suite step.
- [ ] Inspect the YAML and run the equivalent local commands in Task 9.

### Task 9: Focused integration verification and self-review

**Files:**
- Review all changed files and the approved design.

- [ ] Run `python -m compileall -q euclid_polish scripts tests`.
- [ ] Run `ruff check .`.
- [ ] Run `NUMBA_DISABLE_JIT=1 pytest --collect-only -q` as collection-only verification.
- [ ] Run the regression files individually (never as the full suite): `test_web_security.py`, `test_jobs.py`, the selected catalog test module, focused `test_eval_catalog.py`, focused `test_psf_class.py`, `test_sky_lazy_imports.py`, and `test_measure_star_saturation.py`.
- [ ] Review the diff for unsafe Origin parsing, proxy recursion, lock ordering, path/symlink escapes, public export compatibility, serialization compatibility, and accidental unrelated edits.
- [ ] Compare behavior and file coverage to every section of the approved design; add any missing focused assertion before completion.
- [ ] Report focused verification accurately and explicitly state that the full suite was not run at the user's request.
