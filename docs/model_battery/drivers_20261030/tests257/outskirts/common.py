"""Shared pieces for the under-flux investigation (scratch).

tests257 copy: load() also takes ``milestone:member_NNN@STEP`` and opens the milestone's
already-materialized ensemble dir (what milestones.load returns, without re-materializing)."""
import re
import time

import numpy as np
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel
from euclid_polish.eval import knee_attribution as ka
from euclid_polish.training.brightness import (BRIGHTNESS_KERNEL_PX, block_noise_factors,
                                               core_kernel)
from euclid_polish.training.forward_onthefly import member_psf_sets
from euclid_polish.web.helpers import ensemble_viz as ev
from euclid_polish.web.helpers import milestones as ms

BANDS = [Config.get_band(n) for n in Config.HR_TARGET_BAND_NAMES]
SCALES = (1, 2, 4, 8)


def load(label):
    spec = re.fullmatch(r"milestone:(member_\d+)@(\d+)", label)
    if spec:
        member_name, step = spec.group(1), int(spec.group(2))
        ens = ms.local_root() / member_name / "ensembles" / ms.step_dirname(step)
        ckpt = tf.train.latest_checkpoint(str(ens / member_name))
        assert ckpt and ckpt.endswith(f"ckpt-{step}"), (ens, ckpt)
        print("milestone ensemble dir", ens, "checkpoint", ckpt, flush=True)
        model = EnsembleModel(str(ens), labels=[member_name], include_universal=True, starless=False)
    else:
        model = EnsembleModel(ev.ensemble_dir(), labels=[label], include_universal=True, starless=False)
    member = model.members[0]
    return ka.KneeNet.from_member(member), member, model


def kernels():
    sets, note = member_psf_sets(seed=0)
    k = np.stack([core_kernel(sets[b.name].mean().data, BRIGHTNESS_KERNEL_PX) for b in BANDS], -1)
    return tf.constant(k[..., None], tf.float32), note


def block_factors():
    return np.array([block_noise_factors(b, SCALES) for b in BANDS])  # (4, len(SCALES))


def reobserve(img, kernel):
    """(N, H, W, 4) HR electrons → (N, H/2 - 20, W/2 - 20, 4) archive grid (VALID)."""
    blurred = tf.nn.depthwise_conv2d(img, kernel, [1, 1, 1, 1], "VALID")
    return tf.nn.avg_pool2d(blurred, 2, 2, "VALID") * 4.0


def lr_valid(lr):
    """LR pixels matching reobserve's VALID output: 41-px HR kernel trims 20 HR = 10 LR px."""
    h = 10
    return lr[h:-h, h:-h]


def make_fns(net, kernel):
    knees = net.knees

    @tf.function(reduce_retracing=True)
    def forward(x):
        s = ka.asinh_stretch_multi_knee(x, knees)
        sr = net.predict_electrons(s)
        return sr, reobserve(sr, kernel)

    @tf.function(reduce_retracing=True)
    def gain(x, band):
        """∂ Σ A(SR)[..., band] / ∂ x  (all input bands)."""
        with tf.GradientTape() as tape:
            tape.watch(x)
            s = ka.asinh_stretch_multi_knee(x, knees)
            a = reobserve(net.predict_electrons(s), kernel)
            total = tf.reduce_sum(tf.gather(a, band, axis=-1))
        return tape.gradient(total, x)

    return forward, gain


if __name__ == "__main__":
    from euclid_polish.web.helpers import real_tiles as rt
    entries = [e for e in rt.list_entries("nexus") if e.extras.get("jwst_file")]
    lr = np.asarray(rt.get_tile("nexus", entries[0].id, entry=entries[0]).lr_e, np.float32)
    kernel, note = kernels()
    print(note, block_factors())
    for label in ("member_238", "member_249"):
        net, member, model = load(label)
        fwd, gain = make_fns(net, kernel)
        x = tf.constant(lr[None])
        t = time.time(); sr, a = fwd(x); sr = sr.numpy()[0]; print(label, "fwd", time.time() - t)
        t = time.time(); sr, a = fwd(x); print(label, "fwd2", time.time() - t)
        ref = model.member_arrays(lr, indices=[0])[0]
        print("max |KneeNet - served|", float(np.max(np.abs(sr.numpy()[0] - ref))), float(np.max(np.abs(ref))))
        t = time.time(); g = gain(x, tf.constant(0)).numpy()[0]; print("grad", time.time() - t)
        t = time.time(); g = gain(x, tf.constant(1)).numpy()[0]; print("grad2", time.time() - t)
        print("gain Y median interior", np.median(g[40:-40, 40:-40, 1]), "VIS->Y", np.median(g[40:-40, 40:-40, 0]))
