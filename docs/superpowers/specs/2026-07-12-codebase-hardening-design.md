# EuclidPolish Codebase Hardening Design

## Goal

Correct the seven defects confirmed during the July 2026 codebase review, keep
the localhost Web UI frictionless, and establish a minimal automated quality
gate without starting a broad architectural rewrite.

## Scope

This pass includes:

1. same-origin protection for every state-changing Web route;
2. thread-safe capture of concurrent local-job output;
3. repair of the broken test-suite import;
4. lossless `CatalogObject` CSV persistence;
5. consistent evaluation-run selection;
6. PSF provenance preservation during resampling;
7. lazy `euclid_polish.sky` package exports; and
8. a minimal GitHub Actions workflow plus cleanup required to make Ruff pass.

This pass does not add remote Web authentication, redesign the UI, make Pyright
a required gate, split the largest modules, or remove public APIs merely because
Vulture cannot find an in-repository caller.

## Web security boundary

The application remains a zero-login, loopback-only tool. `main()` accepts only
loopback bind addresses (`127.0.0.1`, `::1`, and `localhost`); a non-loopback
`--host` fails before the Flask server starts with an actionable explanation.
Authenticated remote access is a separate feature and is not implied by this
hardening pass.

A single `before_request` guard protects unsafe HTTP methods (`POST`, `PUT`,
`PATCH`, and `DELETE`):

- if `Origin` is present, its scheme and authority must equal the request's
  effective origin;
- if `Sec-Fetch-Site` is present, `cross-site` is rejected;
- same-origin browser requests continue unchanged; and
- non-browser localhost clients that send neither browser header continue to
  work, preserving scripts and the existing Flask test suite.

The guard is registered before route-specific work and returns a small JSON 403
for API/XHR routes or a plain 403 otherwise. The security tests patch a harmless
mutation function and prove that a hostile origin cannot reach it while a
same-origin request can.

## Concurrent job output

`contextlib.redirect_stdout` and `redirect_stderr` are removed from job threads.
Instead, `jobs.py` installs one stream proxy per process for stdout and stderr.
Each proxy uses thread-local state:

- a job thread temporarily binds its `StringIO` buffer;
- writes from that thread go to that buffer;
- writes from every other thread go to the original stream; and
- exiting a job clears only that thread's binding.

This preserves output from existing `print()`-based jobs without editing every
job target. `Job` owns a lock protecting log writes, log reads, progress updates,
and serialization so polling cannot race a writer. A coordinated two-job test
must prove that `A2` remains in job A's log rather than appearing in job B's.

## Catalog persistence

`kind` becomes a base CSV column and participates in `to_row` / `from_row`.
Legacy files without the column load as `kind="star"`. Empty or non-string kind
values also fall back to `star`.

Coordinate deserialization uses explicit `is None` checks rather than boolean
fallback, so `ra=0.0` and `dec=0.0` remain valid. Regression tests cover both the
in-memory row round trip and the full CSV write/read path.

## Evaluation run selection

`evaluation.py` gains one resolver that maps:

- empty, `eval_results`, or omitted run names to `Config.EVAL_RESULTS_DIR`; and
- a simple discovered run name to its direct child directory.

Separators, dot segments, symlink escapes, missing directories, and directories
without the required manifest/data file are rejected with the route-appropriate
400 or 404 response. The resolver replaces ad hoc root selection across run
summary, rerender, morphology, morphology embedding, transformation,
lens-finder summary, and other endpoints that already accept `run`.

The response reports the selected run name, and tests seed different root and
child manifests to prove that `?run=child` returns child data.

## PSF provenance

`PSF.resampled_to()` uses `dataclasses.replace` rather than constructing an
unrelated PSF. Pixel data and scale change while the stamp and remaining
metadata survive, matching the copy-on-write behavior of cropping, recentering,
rotation, and background cleaning. The no-op path continues returning `self`.

## Lazy sky exports

`euclid_polish.sky` and `euclid_polish.sky.generation` stop eagerly importing
the complete generation stack. Their existing public names remain available
through module-level `__getattr__` lazy resolution and accurate `__all__`
metadata. Direct imports from concrete modules are unchanged.

Import-regression tests start a clean Python process, import an observation-only
module, and assert that `lenstronomy` and the lens-generation module are absent
from `sys.modules`. Separate compatibility tests access representative legacy
re-exports and confirm they resolve correctly.

## Test repair and CI

The saturation diagnostic test imports the canonical
`adu_per_s_to_electrons_factor` from `euclid_polish.photometry`; the removed
private duplicate is not restored.

The current Ruff findings are cleaned so CI begins green. Mechanical import and
format fixes must not change behavior; unsafe fixes are reviewed individually.
Pyright remains informational because the repository currently has 302 errors
under its configured standard mode.

GitHub Actions uses the repository's Conda environment on Linux and runs:

1. `python -m compileall -q euclid_polish scripts tests`;
2. `ruff check .`;
3. test collection with `NUMBA_DISABLE_JIT=1`; and
4. the focused regression suites for security, jobs, catalog persistence,
   evaluation routing, PSF provenance, and import boundaries.

The workflow does not claim full-suite coverage. The full suite remains a local
or dedicated-runner task until its memory requirements and external-data lanes
are separated reliably.

## Error handling and compatibility

All new validation fails closed at the boundary and returns existing Flask-style
responses. Existing localhost URLs, same-origin forms, fetch calls, CLI scripts,
CSV files, public sky imports, and unstamped PSFs remain supported. No new runtime
dependency is introduced.

## Verification

Each behavior change follows red-green-refactor:

1. add one minimal regression test;
2. run it and confirm the reviewed defect causes the expected failure;
3. implement the smallest correction;
4. rerun the focused test until green; and
5. run the accumulated focused suite, compilation, Ruff, and CI-equivalent
   collection checks before completion.

The final report must distinguish focused verification from a full-suite pass
and must report any environment or memory limitation exactly.
