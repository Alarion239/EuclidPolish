#!/usr/bin/env python3
"""
Visualize PSF (Point Spread Function) data from Euclid256 dataset.
Displays PSF in both normal and log scales.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.gridspec as gridspec


def load_psf_files(psf_dir, n_images=16):
    """
    Load PSF files from directory.

    Parameters
    ----------
    psf_dir : str or Path
        Directory containing PSF .npy files
    n_images : int
        Number of PSF images to load

    Returns
    -------
    psf_data : list of ndarray
        List of PSF arrays
    filenames : list of str
        Corresponding filenames
    """
    psf_dir = Path(psf_dir)
    psf_files = sorted(psf_dir.glob("*.npy"))[:n_images]

    psf_data = []
    filenames = []

    for psf_file in psf_files:
        psf = np.load(psf_file)
        psf_data.append(psf)
        filenames.append(psf_file.name)

    return psf_data, filenames


def plot_single_psf(psf, title="PSF", save_path=None):
    """
    Plot a single PSF in normal and log scale side by side.

    Parameters
    ----------
    psf : ndarray
        PSF data array
    title : str
        Title for the plot
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Normal scale
    im1 = axes[0].imshow(psf, origin='lower', cmap='viridis')
    axes[0].set_title(f'{title} - Normal Scale')
    axes[0].set_xlabel('x (pixels)')
    axes[0].set_ylabel('y (pixels)')
    plt.colorbar(im1, ax=axes[0], label='Intensity')

    # Log scale (add small epsilon to avoid log(0))
    epsilon = 1e-10
    psf_log = np.log10(psf + epsilon)
    im2 = axes[1].imshow(psf_log, origin='lower', cmap='viridis')
    axes[1].set_title(f'{title} - Log Scale')
    axes[1].set_xlabel('x (pixels)')
    axes[1].set_ylabel('y (pixels)')
    plt.colorbar(im2, ax=axes[1], label='Log10(Intensity)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_psf_grid(psf_data, filenames, n_cols=4, save_path=None):
    """
    Plot multiple PSFs in a grid with normal and log scales.

    Parameters
    ----------
    psf_data : list of ndarray
        List of PSF arrays
    filenames : list of str
        List of filenames for titles
    n_cols : int
        Number of columns in the grid
    save_path : str, optional
        Path to save the figure
    """
    n_images = len(psf_data)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows * 2))
    gs = gridspec.GridSpec(n_rows * 2, n_cols, hspace=0.3, wspace=0.3)

    epsilon = 1e-10

    for idx, (psf, filename) in enumerate(zip(psf_data, filenames)):
        row = idx // n_cols
        col = idx % n_cols

        # Normal scale
        ax1 = fig.add_subplot(gs[2 * row, col])
        im1 = ax1.imshow(psf, origin='lower', cmap='viridis')
        ax1.set_title(f'{filename}\nNormal', fontsize=8)
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Log scale
        ax2 = fig.add_subplot(gs[2 * row + 1, col])
        psf_log = np.log10(psf + epsilon)
        im2 = ax2.imshow(psf_log, origin='lower', cmap='viridis')
        ax2.set_title(f'{filename}\nLog Scale', fontsize=8)
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_psf_statistics(psf_dir, save_path=None):
    """
    Plot statistics across all PSF files in directory.

    Parameters
    ----------
    psf_dir : str or Path
        Directory containing PSF .npy files
    save_path : str, optional
        Path to save the figure
    """
    psf_dir = Path(psf_dir)
    psf_files = sorted(psf_dir.glob("*.npy"))

    # Compute statistics
    max_values = []
    total_flux = []
    centers = []

    for psf_file in psf_files:
        psf = np.load(psf_file)
        max_values.append(psf.max())
        total_flux.append(psf.sum())

        # Compute centroid
        y, x = np.indices(psf.shape)
        total = psf.sum()
        if total > 0:
            x_center = (x * psf).sum() / total
            y_center = (y * psf).sum() / total
            centers.append((x_center, y_center))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Max values
    axes[0, 0].plot(max_values, 'o-', markersize=3)
    axes[0, 0].set_title('Maximum Intensity per PSF')
    axes[0, 0].set_xlabel('PSF Index')
    axes[0, 0].set_ylabel('Max Intensity')
    axes[0, 0].grid(True, alpha=0.3)

    # Total flux
    axes[0, 1].plot(total_flux, 'o-', markersize=3, color='orange')
    axes[0, 1].set_title('Total Flux per PSF')
    axes[0, 1].set_xlabel('PSF Index')
    axes[0, 1].set_ylabel('Total Flux')
    axes[0, 1].grid(True, alpha=0.3)

    # Centroid X positions
    x_centers = [c[0] for c in centers]
    axes[1, 0].plot(x_centers, 'o-', markersize=3, color='green')
    axes[1, 0].set_title('Centroid X Position')
    axes[1, 0].set_xlabel('PSF Index')
    axes[1, 0].set_ylabel('X Position (pixels)')
    axes[1, 0].axhline(y=32, color='r', linestyle='--', label='Center (32)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Centroid Y positions
    y_centers = [c[1] for c in centers]
    axes[1, 1].plot(y_centers, 'o-', markersize=3, color='red')
    axes[1, 1].set_title('Centroid Y Position')
    axes[1, 1].set_xlabel('PSF Index')
    axes[1, 1].set_ylabel('Y Position (pixels)')
    axes[1, 1].axhline(y=32, color='r', linestyle='--', label='Center (32)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def main():
    """Main function to run PSF visualization."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize PSF data')
    parser.add_argument('--psf-dir', type=str, default='euclid256/psf',
                        help='Directory containing PSF files')
    parser.add_argument('--n-images', type=int, default=16,
                        help='Number of PSF images to visualize')
    parser.add_argument('--mode', type=str, choices=['single', 'grid', 'stats', 'all'],
                        default='all', help='Visualization mode')
    parser.add_argument('--index', type=int, default=0,
                        help='Index of single PSF to plot (for single mode)')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save figure(s)')

    args = parser.parse_args()

    if args.mode in ['single', 'all']:
        psf_data, filenames = load_psf_files(args.psf_dir, n_images=1)
        if psf_data:
            idx = min(args.index, len(psf_data) - 1)
            psf_files = sorted(Path(args.psf_dir).glob("*.npy"))
            psf = np.load(psf_files[idx])
            save_path = f"{args.save}_single.png" if args.save else None
            plot_single_psf(psf, title=psf_files[idx].name, save_path=save_path)

    if args.mode in ['grid', 'all']:
        psf_data, filenames = load_psf_files(args.psf_dir, n_images=args.n_images)
        save_path = f"{args.save}_grid.png" if args.save else None
        plot_psf_grid(psf_data, filenames, n_cols=4, save_path=save_path)

    if args.mode in ['stats', 'all']:
        save_path = f"{args.save}_stats.png" if args.save else None
        plot_psf_statistics(args.psf_dir, save_path=save_path)


if __name__ == "__main__":
    main()
