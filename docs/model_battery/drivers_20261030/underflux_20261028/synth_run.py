"""Synthetic test fields: LR, truth (target-PSF HR, starfull + starless) and both models' SR.

python synth_run.py OUT_DIR {fields|gains}
fields: lr_syn.npy (100, 255, 255, 4); RT_syn / RC_syn = 2×2-summed targets (starfull / starless);
        R_syn_<label> = 2×2-summed SR; K_syn = unit-sum 21×21 LR kernel fitted so K ⊛ RT ≈ LR
        (the truth-side operator, so A = K ⊛ R needs no ePSF guess); A_syn_<label>, AT_syn (235² valid).
gains:  G_syn_<label> (NG, 4, 255, 255, 4) = ∂ Σ A_meanPSF(SR)_b / ∂ LR for the first NG fields.
"""
import sys

import numpy as np
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

from common import block_factors, kernels, load, make_fns
from euclid_polish.config import Config
from euclid_polish.eval import knee_attribution as ka
from euclid_polish.image.collection import ImageSet
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.training.target_blur import blur_target_array
from euclid_polish.web.helpers import ensemble_viz as ev

out_dir, phase = sys.argv[1], sys.argv[2]
MODELS = ("member_238", "member_249")
H = 10
NG = 30


def rebin(x):
    return x.reshape(255, 2, 255, 2, 4).sum((1, 3))


if phase == "fields":
    d = ev._sky_records_local_dir()
    fwhm = float(Config.TARGET_PSF_FWHM_ARCSEC)
    lr_all = np.lib.format.open_memmap(f"{out_dir}/lr_syn.npy", "w+", np.float32, (100, 255, 255, 4))
    RT = np.lib.format.open_memmap(f"{out_dir}/RT_syn.npy", "w+", np.float32, (100, 255, 255, 4))
    RC = np.lib.format.open_memmap(f"{out_dir}/RC_syn.npy", "w+", np.float32, (100, 255, 255, 4))
    streams = [ImageSet.read(tfrecord_path(d, k)) for k in ("dirty_test", "hr_test", "clean_test")]
    for i, (lr, hr, clean) in enumerate(zip(*streams)):
        assert lr.index == hr.index == clean.index == i
        lr_all[i] = lr.data
        for dst, im in ((RT, hr), (RC, clean)):
            t = blur_target_array(np.asarray(im.data, np.float32), fwhm, pixel_scale_arcsec=im.pixel_scale_arcsec)
            dst[i] = rebin(np.asarray(t, np.float32))
    for a in (lr_all, RT, RC):
        a.flush()
    print("fields loaded", flush=True)

    # truth-side LR kernel: LR ≈ K ⊛ RT  (ΣK = 1, inverse-variance LSQ, centre-substituted)
    factors = block_factors()
    p = (2 * H + 1) ** 2
    c = p // 2
    XtX = np.zeros((4, p - 1, p - 1))
    Xty = np.zeros((4, p - 1))
    for i in range(100):
        lr = np.asarray(lr_all[i], np.float64)
        r = np.asarray(RT[i], np.float64)
        sky = ka.noise_sigma(lr) / factors[:, 0]
        for b in range(4):
            win = sliding_window_view(r[..., b], (2 * H + 1, 2 * H + 1)).reshape(-1, p)
            y = lr[H:-H, H:-H, b].ravel()
            ok = np.isfinite(y) & (y != 0)
            w = 1.0 / (sky[b] ** 2 + np.clip(y, 0, None))
            X = (np.delete(win, c, 1) - win[:, c:c + 1])[ok]
            t = (y - win[:, c])[ok]
            XtX[b] += X.T @ (X * w[ok][:, None])
            Xty[b] += X.T @ (t * w[ok])
    K = np.zeros((4, 2 * H + 1, 2 * H + 1))
    for b in range(4):
        k = np.linalg.solve(XtX[b], Xty[b])
        K[b] = np.insert(k, c, 1.0 - k.sum()).reshape(2 * H + 1, 2 * H + 1)
    np.save(f"{out_dir}/K_syn.npy", K)
    print("K_syn centre", K[:, H, H].round(3), "frac r<=2",
          [round(float(K[b, H - 2:H + 3, H - 2:H + 3].sum()), 3) for b in range(4)], flush=True)

    def observe(r):
        return np.stack([ndimage.correlate(r[..., b], K[b], mode="constant") for b in range(4)], -1)[H:-H, H:-H]

    AT = np.lib.format.open_memmap(f"{out_dir}/AT_syn.npy", "w+", np.float32, (100, 235, 235, 4))
    for i in range(100):
        AT[i] = observe(np.asarray(RT[i], np.float64))
    AT.flush()
    kernel, _ = kernels()
    for label in MODELS:
        net, _, _ = load(label)
        fwd, _ = make_fns(net, kernel)
        R = np.lib.format.open_memmap(f"{out_dir}/R_syn_{label}.npy", "w+", np.float32, (100, 255, 255, 4))
        A = np.lib.format.open_memmap(f"{out_dir}/A_syn_{label}.npy", "w+", np.float32, (100, 235, 235, 4))
        for i in range(100):
            sr, _ = fwd(tf.constant(lr_all[i][None]))
            R[i] = rebin(sr.numpy()[0])
            A[i] = observe(np.asarray(R[i], np.float64))
        R.flush()
        A.flush()
        print(label, "SR done", flush=True)

elif phase == "gains":
    lr_all = np.load(f"{out_dir}/lr_syn.npy", mmap_mode="r")
    kernel, _ = kernels()
    for label in MODELS:
        net, _, _ = load(label)
        _, gain = make_fns(net, kernel)
        G = np.lib.format.open_memmap(f"{out_dir}/G_syn_{label}.npy", "w+", np.float32, (NG, 4, 255, 255, 4))
        for j in range(NG):
            x = tf.constant(lr_all[j][None])
            for b in range(4):
                G[j, b] = gain(x, tf.constant(b)).numpy()[0]
            print(label, "gain", j, flush=True)
        G.flush()
print("done", flush=True)
