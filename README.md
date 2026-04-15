# EuclidPolish

We use the POLISH algorithm (https://arxiv.org/abs/2111.03249) on Euclid data.

The super-resolution neural network and the code to train it comes from the WDSR implementation found here:
https://github.com/krasserm/super-resolution

## Pipeline

### 1. Generate clean HR sky images

Use GalSim with the COSMOS parametric catalog to simulate fields at the HR pixel scale (0.05 arcsec/pixel). Each 256x256 image contains Poisson-sampled galaxies (40/arcmin^2) and stars (2/arcmin^2). Stars are drawn as point sources with magnitudes sampled from a 3-bin distribution (faint/mid/bright). Galaxies are drawn with `method="no_pixel"`.

Output: `clean_train.tfrecord`, `clean_validate.tfrecord` (raw flux, float32).

*polish-pub equivalent: `simulation.py` `SimRadioGal.sim_sky()` — radio galaxies as 2D elliptical Gaussians at 0.25 arcsec/pixel.*

### 2. Extract PSF

Build an empirical PSF (ePSF) from real Euclid VIS cutouts using `photutils.psf.EPSFBuilder`. Cutouts are downloaded from the Euclid archive, validated, then stacked.

*polish-pub equivalent: uses a pre-computed PSF FITS file (e.g. `dsa-2000-fullband-psf.fits`) or a synthetic Gaussian kernel.*

### 3. Generate dirty LR images

For each clean HR image, matching polish-pub's `convolvehr()` + `create_LR_image()` pipeline:

1. Add Gaussian noise: `std = noise_fraction * mean(image)` (default `noise_fraction = 0.1`)
2. Convolve with PSF via FFT: `scipy.signal.fftconvolve(..., mode='same')` at full resolution
3. `normalize_data` full-res convolved → uint16 [0, 65535]
4. Stride downsample: `data[N//2::N, N//2::N]` (default `N = 2`) — on uint16 data
5. `normalize_data` downsampled → uint16 [0, 65535] (re-stretches range after downsample)
6. `normalize_data` noise-free clean HR → uint16 [0, 65535]

`normalize_data` matches polish-pub exactly: `data - min`, `data / max`, `* 65535`, cast to uint16.

Output: `dirty_{subset}_norm.tfrecord` and `clean_{subset}_norm.tfrecord` (uint16-valued float32, [0, 65535]). Raw float32 clean HR and raw dirty LR are also written for inspection.

*polish-pub difference: fixed noise std=5 instead of fraction-of-mean; default rebin=4 (ours: 2). Saves as uint16 PNG pairs instead of TFRecords.*

### 4. Train

Model: WDSR-B (`scripts/polish-pub/model/wdsr.py` `wdsr_b`). 32 residual blocks, 32 filters, expansion factor 6, weight normalization, pixel-shuffle upsampling. Internal normalization: `(x - 32768) / 32768` on input, inverse on output.

Data: load `*_norm` TFRecords (already [0, 65535]). Augmentation is random crop only (96x96 HR patches) — rotation and flip are disabled because the Euclid PSF is asymmetric.

Loss: MAE. Optimizer: Adam, lr=1e-3 decaying to 5e-4 at step 200k. Checkpoints: save best 3 by PSNR.

*polish-pub differences: uses random crop + random 90° rotation + random flip for augmentation; default batch size 4 (ours: 16); default scale 4 (ours: 2); `tensorflow_addons` WeightNormalization (ours: `tensorflow_probability`). Also has EDSR, SRGAN, and WDSR-A architectures — we only use WDSR-B.*

### 5. Inference

Load model from checkpoint or `.h5` weights. Input must already be in [0, 65535]. Feed directly to model, clip output to [0, 65535]. No per-image normalization at inference time.

*polish-pub equivalent: `reconstruct.py` — same, but `resolve_single` rounds and casts to uint16. Ours returns float32.*

## Key differences from polish-pub

| | polish-pub | EuclidPolish |
|---|---|---|
| Domain | Radio (DSA-2000) | Optical (Euclid VIS) |
| Sky simulation | 2D Gaussian/Sersic galaxies | GalSim COSMOS parametric |
| Pixel scale | 0.25 arcsec/pixel | 0.05 arcsec/pixel |
| PSF | Pre-computed / Gaussian | Empirical ePSF from real cutouts |
| Noise std | Fixed (5) | 10% of per-image mean flux |
| Default scale | 4x | 2x |
| Normalization order | normalize full-res → downsample → normalize again | Same (matching polish-pub) |
| Data format | uint16 PNG + TF cache | TFRecord (uint16-valued float32) |
| Augmentation | Crop + rotate + flip | Crop only (asymmetric PSF) |
| Weight norm lib | tensorflow_addons | tensorflow_probability |
| Inference output | uint16 (rounded) | float32 (clipped) |
