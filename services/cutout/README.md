# Euclid Q1 cutout server and POLISH viewer

Returns all four MER bands (VIS, Y_E, J_E, H_E) of a pixel window of a Q1 MER
tile in one response, for the in-browser super-resolution viewer. It reads the
public `nasa-irsa-euclid-q1` bucket with HTTP range requests, one per image
row, so it belongs next to the bucket (us-east-1). API and binary format are
documented at the top of [cutout_server.py](cutout_server.py).

The same service serves the viewer at `/` ([static/](static/)): the page
fetches 320² blocks (256² kept + 32 px margins), runs member 257 on them with
ONNX Runtime Web 1.31.1 (WebGPU, else multi-threaded WebAssembly; the page is
cross-origin isolated for that), and shows Euclid against POLISH with a swipe
divider. `static/m257.onnx` comes from `scripts/export_onnx.py`;
`static/tiles.json` is the project's Q1 tile index (a test keeps them equal).
Far out, the page is a map: Aladin Lite 3.8.2 showing Euclid's own HiPS
(CDS Q1 colour when CDS answers, else ESA's VIS FITS HiPS) with the Q1 tile
outlines, and no cutouts or model. Zooming in to 0.1″ per screen pixel inside
the outlines switches to the POLISH view (the model loads then); zooming out
past 0.125″/px switches back. The page is cross-origin isolated with
COEP `credentialless`, since the HiPS servers send CORS but not CORP.
Checked 2026-11-18: the browser's SR of real blocks equals TensorFlow's
(block sums to 6 decimals); ~210–300 ms per block on an M4 with WebGPU.

```bash
docker build -t euclid-cutout services/cutout
docker run --rm -p 8080:8080 euclid-cutout
curl -o cut.bin "localhost:8080/v1/cutout?tile=102158892&x0=9000&y0=9000&size=320"
```

Railway: deployed 2026-11-18 as project `euclid-cutout`, service `cutout`,
at https://cutout-production-538f5738.up.railway.app (`railway up` from this
directory). `railway.toml` picks the Dockerfile, the `/health` check and the
US East (Virginia) region, next to the bucket. Variables: `S3_CONCURRENCY=256`
is set there (S3 answers each range request in ~20–30 ms, so a 320² 4-band
block, 1,280 row reads, takes ~0.57 s at 64 and ~0.22–0.33 s at 256 once the
tile's headers are cached; the first block of a tile ~0.85 s). Optional:
`ALLOWED_ORIGINS`, `MAX_CUTOUT`.

Checked 2026-11-18, locally and on Railway: a 321² VIS cutout is bit-identical to IRSA's cutout
service. Tests: `tests/test_cutout_service.py` (fake bucket, synthetic FITS).
