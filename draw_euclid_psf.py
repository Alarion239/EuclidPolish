#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import galsim
import euclidlike


VALID_BANDS = {
    "VIS": "VIS",
    "NISP_H": "H",
    "NISP_J": "J",
    "NISP_Y": "Y",
}

LAMBDA_EFF_NM = {
    "VIS": 677.0,
    "NISP_H": 1830.0,
    "NISP_J": 1250.0,
    "NISP_Y": 990.0,
}


def radial_profile(image, center=None, nbins=None):
    arr = np.asarray(image, dtype=float)
    ny, nx = arr.shape

    if center is None:
        cx = (nx - 1) / 2.0
        cy = (ny - 1) / 2.0
    else:
        cx, cy = center

    y, x = np.indices(arr.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    if nbins is None:
        nbins = min(nx, ny) // 2

    rmax = r.max()
    edges = np.linspace(0.0, rmax, nbins + 1)
    prof = np.zeros(nbins, dtype=float)
    counts = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        counts[i] = np.count_nonzero(m)
        if counts[i] > 0:
            prof[i] = arr[m].mean()

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, prof


def estimate_fwhm_pixels(image):
    arr = np.asarray(image, dtype=float)
    ny, nx = arr.shape
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0

    r, prof = radial_profile(arr, center=(cx, cy), nbins=min(nx, ny) // 2)

    peak = prof.max()
    half = peak / 2.0

    below = np.where(prof <= half)[0]
    if len(below) == 0:
        return np.nan

    i = below[0]
    if i == 0:
        r_half = r[0]
    else:
        x0, y0 = r[i - 1], prof[i - 1]
        x1, y1 = r[i], prof[i]
        if y1 == y0:
            r_half = x1
        else:
            r_half = x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    return 2.0 * r_half


def main():
    parser = argparse.ArgumentParser(description="Render and inspect a Euclid PSF.")
    parser.add_argument("--psf-dir", required=True, help="Directory containing euclidlike PSF FITS files")
    parser.add_argument("--band", default="VIS", choices=list(VALID_BANDS.keys()))
    parser.add_argument("--ccd", type=int, default=0)
    parser.add_argument("--x", type=float, default=None, help="CCD x position in pixels")
    parser.add_argument("--y", type=float, default=None, help="CCD y position in pixels")
    parser.add_argument("--pixel-scale", type=float, default=0.1, help="Render scale in arcsec/pixel")
    parser.add_argument("--stamp-size", type=int, default=129, help="PSF stamp size in pixels")
    parser.add_argument("--output", default="euclid_psf.png", help="Output figure path")
    parser.add_argument("--save-npy", default=None, help="Optional path to save PSF array as .npy")
    args = parser.parse_args()

    if not os.path.isdir(args.psf_dir):
        raise FileNotFoundError(f"PSF directory does not exist: {args.psf_dir}")

    if args.x is None:
        x = euclidlike.n_pix_col / 2
    else:
        x = args.x

    if args.y is None:
        y = euclidlike.n_pix_row / 2
    else:
        y = args.y

    pos = galsim.PositionD(x=x, y=y)

    psf = euclidlike.getPSF(
        ccd=args.ccd,
        bandpass=VALID_BANDS[args.band],
        ccd_pos=pos,
        wavelength=LAMBDA_EFF_NM[args.band],
        psf_dir=args.psf_dir,
    )

    stamp = galsim.ImageD(args.stamp_size, args.stamp_size, scale=args.pixel_scale)
    psf.drawImage(image=stamp)
    arr = stamp.array

    arr_sum = arr.sum()
    arr_peak = arr.max()
    fwhm_pix = estimate_fwhm_pixels(arr)
    fwhm_arcsec = fwhm_pix * args.pixel_scale if np.isfinite(fwhm_pix) else np.nan

    print(f"Band: {args.band}")
    print(f"CCD: {args.ccd}")
    print(f"CCD position: ({x:.2f}, {y:.2f})")
    print(f"Render pixel scale: {args.pixel_scale:.6f} arcsec/pixel")
    print(f"Stamp size: {args.stamp_size} px")
    print(f"PSF sum: {arr_sum:.8f}")
    print(f"PSF peak: {arr_peak:.8e}")
    print(f"Estimated FWHM: {fwhm_pix:.3f} px = {fwhm_arcsec:.3f} arcsec")

    if args.save_npy:
        np.save(args.save_npy, arr)

    r, prof = radial_profile(arr)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].imshow(arr, origin="lower")
    axes[0].set_title("PSF")
    axes[0].set_xlabel("x [pix]")
    axes[0].set_ylabel("y [pix]")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(np.log10(arr + 1e-12), origin="lower")
    axes[1].set_title("log10(PSF)")
    axes[1].set_xlabel("x [pix]")
    axes[1].set_ylabel("y [pix]")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].plot(r, prof)
    axes[2].axhline(prof.max() / 2.0, linestyle="--")
    axes[2].set_title("Radial profile")
    axes[2].set_xlabel("radius [pix]")
    axes[2].set_ylabel("mean intensity")

    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
