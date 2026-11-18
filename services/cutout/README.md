# Euclid Q1 cutout server

Returns all four MER bands (VIS, Y_E, J_E, H_E) of a pixel window of a Q1 MER
tile in one response, for the in-browser super-resolution viewer. It reads the
public `nasa-irsa-euclid-q1` bucket with HTTP range requests, one per image
row, so it belongs next to the bucket (us-east-1). API and binary format are
documented at the top of [cutout_server.py](cutout_server.py).

```bash
docker build -t euclid-cutout services/cutout
docker run --rm -p 8080:8080 euclid-cutout
curl -o cut.bin "localhost:8080/v1/cutout?tile=102158892&x0=9000&y0=9000&size=320"
```

Railway: new service from this repo, Root Directory `services/cutout`
(`railway.toml` picks the Dockerfile and the `/health` check), region
US East (Virginia). Optional variables: `ALLOWED_ORIGINS`, `S3_CONCURRENCY`,
`MAX_CUTOUT`.

Checked 2026-11-18: a 321² VIS cutout is bit-identical to IRSA's cutout
service. Tests: `tests/test_cutout_service.py` (fake bucket, synthetic FITS).
