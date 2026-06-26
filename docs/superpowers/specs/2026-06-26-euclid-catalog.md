# EuclidCatalog — one authenticated client for Euclid query + cutout download

## Context

`euclid_polish/euclid/` does three unrelated jobs: (1) query the Euclid archive + download cutouts, (2) PSF extraction/loading, (3) read evaluation catalogs. Underneath, everything relies on astroquery's process-global `Euclid` singleton, and authentication is ambient module-level state (`auth.py` globals) that callers must remember to prime. There is no object that *owns* a session.

This change makes the catalog/download spine a single authenticated client and evicts the unrelated concerns.

## Goals / non-goals

- **Goal:** `euclid_polish/catalog/` package built around one class, `EuclidCatalog`, that authenticates on construction and exposes the two real features — query objects, download cutouts — as methods.
- **Goal:** PSF code and evaluation-catalog code move to their real homes (`psf/`, `eval/`).
- **Non-goal:** changing the physics, the ADQL, the photometric conversions, or the on-disk CSV/FITS formats. This is structure + naming + auth lifecycle.
- **Non-goal:** backward compatibility. We regenerate freely and update every call site; no aliases, no shims.

## Authentication (the headline change)

`EuclidCatalog.__init__` authenticates eagerly. Construction either returns an authenticated client or raises.

```python
EuclidCatalog(login=None, password=None)
#  login & password passed                  -> use them
#  else EUCLID_USER / EUCLID_PASSWORD set    -> use those
#  else                                      -> raise EuclidAuthError
#  then Euclid.login(...); on failure        -> raise EuclidAuthError
```

- **No interactive prompt, no credentials-file path.** Today's `auth.login()` 3-way precedence (env → file → interactive) collapses to env-or-explicit. `login_interactive`, `login_with_file`, and the file precedence are deleted.
- The instance owns the session lifecycle: the re-login-on-expiry retry that `downloader` does today becomes a private `EuclidCatalog` method, serialised by a class-level lock (astroquery's singleton is still process-global underneath; the lock lives on the class, not a module global).
- **Test/offline seam:** an internal construction path that skips the network login (e.g. `EuclidCatalog._unauthenticated()` / injected session) so the suite builds a client without credentials. The *public* contract stays env-or-explicit-or-error.

## The `EuclidCatalog` surface (absorbs StarCatalog + galaxy_catalog + downloader + archive)

```python
cat = EuclidCatalog()                                  # auth here

# query
stars    = cat.query_bright_stars(region, ...)         # was StarCatalog.query_brightest_stars
galaxies = cat.query_galaxies(region, ...)             # was galaxy_catalog.galaxy_adql + fetch
rows     = cat.query(adql)                              # low-level escape hatch

# download
img  = cat.fetch_image(ra, dec, size)                  # was EuclidArchive.fetch -> Image (4-band)
path = cat.fetch_cutout(ra, dec, size, band)           # was downloader.fetch_cutout_at (one band)
cat.download_cutouts(stars, out_dir, ...)              # was EuclidCutoutDownloader.download (batch)
```

`StarCatalog` is removed. Its non-query responsibilities become module-level functions in `catalog/`:
- on-disk star dataset: `save_star_catalog(stars, path)` / `load_star_catalog(path)` (keeps the stable provenance id + `stars.csv.prov.json` sidecar behaviour).
- per-band quality predicates used by renderers: `star_is_valid(star, band)` / `star_is_corrupted(star, band)`.

## What moves where

| File today (`euclid/`) | Destination | Becomes |
|---|---|---|
| `auth.py` | folded into `catalog/client.py` | `EuclidCatalog.__init__` + private session mgmt |
| `catalog.py` (`StarCatalog`) | `catalog/` | query → `EuclidCatalog`; persist/predicates → module functions |
| `galaxy_catalog.py` | folded into `EuclidCatalog` | `query_galaxies` |
| `downloader.py` | `catalog/download.py` + `EuclidCatalog` | mechanics stay; `EuclidCatalog` is the entry |
| `archive.py` | folded into `EuclidCatalog` | `fetch_image` |
| `photometry.py` | `catalog/photometry.py` | unchanged (utility) |
| `validator.py` | `catalog/validator.py` | unchanged (utility) |
| `cutout_integrity.py` | `catalog/cutout_integrity.py` | unchanged |
| `types.py` (PSF shim) | **deleted** | callers import `euclid_polish.psf` |
| `psf_extractor.py` | `euclid_polish/psf/` | unchanged |
| `psf_library.py` | `euclid_polish/psf/` | unchanged |
| `eval_catalog.py` | `euclid_polish/eval/` | unchanged |
| `lens_catalog.py` | `euclid_polish/eval/` | unchanged (Zenodo lens ground-truth) |

## Phasing

1. **Evict the misplaced concerns** (mechanical, low-risk, independent of auth): move `psf_extractor`/`psf_library` → `psf/`, `eval_catalog`/`lens_catalog` → `eval/`, delete the `types.py` shim. Update call sites.
2. **Rename the package** `euclid_polish/euclid/` → `euclid_polish/catalog/`; update imports.
3. **Build `EuclidCatalog`**: auth-on-construction (the contract above), absorb the query/download/persist surfaces, delete `auth.py` globals and `StarCatalog` the class, migrate the ~8 external callers + tests.

Each phase lands green and auto-commits on the branch.

## Testing

- New `tests/test_euclid_catalog.py`: `EuclidCatalog()` with no env + no args raises `EuclidAuthError`; with env vars set (login mocked) constructs; explicit `login=/password=` path; query/download methods delegate correctly (astroquery mocked). The offline seam builds a client for the other tests.
- Existing PSF / photometry / downloader / eval-catalog suites move/rename with their modules and stay green.

## Risks

- **Eager network login in `__init__`** — mitigated by the offline construction seam; every test that needs a client uses it.
- **Process-global astroquery singleton** — two `EuclidCatalog` instances share one underlying session; the class-level lock serialises (re)login. Documented, not hidden.
- **Blast radius** — ~50 import sites across ~34 files; phased so each step is independently green.
