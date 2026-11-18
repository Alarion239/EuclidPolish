"""Export a universal member to ONNX for the in-browser viewer (services/cutout/static).

    PYTHONPATH=. python scripts/export_onnx.py ckpt/ensemble/member_257 \
        services/cutout/static/m257.onnx

tf2onnx, onnx and onnxruntime are not in the project env; install them anywhere
on the path, e.g. ``pip install --target /tmp/onnxlib tf2onnx onnx onnxruntime``
and prepend that directory to ``PYTHONPATH``.

The graph takes ``lr``: ``(1, H, W, 32)`` float32, the four bands' electrons
stretched as ``asinh(e / q)`` at the member's 8 input knees, knee-major
(channel ``k·4 + b``), and returns ``(1, 2H, 2W, 4)`` electrons per SR pixel
(the learned output knee is inside the graph).

Two export details:
* tfp's ``WeightNorm`` initialises lazily behind a ``tf.cond`` that assigns a
  variable, which tf2onnx's grappler pass rejects. The member is rebuilt with
  ``eager_weight_norm_init=True`` (same variables, bit-identical outputs, checked
  below) so the graph holds only the normalised kernels.
* tf2onnx leaves the folded weight norms' ``reduction_indices`` constants as
  unused graph inputs; they are dropped.

The script checks the ONNX model against TensorFlow on a stretched synthetic
field and fails if any pixel differs by more than 1e-3 relative.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx

from euclid_polish.training.inference import (
    infer_checkpoint_asinh_knees,
    infer_checkpoint_num_res_blocks,
    load_model_from_checkpoint,
)
from euclid_polish.training.models.output_knee import LearnedOutputKnee
from euclid_polish.training.models.wdsr import initialize_weight_norms, wdsr


def plain_graph_model(checkpoint_dir: str):
    original = load_model_from_checkpoint(checkpoint_dir, scale=2)
    initialize_weight_norms(original)
    heads = [layer for layer in original.layers if isinstance(layer, LearnedOutputKnee)]
    if len(heads) != 1:
        raise SystemExit("export_onnx: only members with a learned output knee are supported")
    head = heads[0]
    n_in = int(original.input_shape[-1])
    n_out = int(original.output_shape[-1])
    blocks = infer_checkpoint_num_res_blocks(checkpoint_dir)
    model = wdsr(scale=2, num_res_blocks=blocks, nchan_in=n_in, nchan_out=n_out,
                 input_knees=n_in // n_out, learned_output_knee=head.init_knee_e,
                 learned_knee_bounds=(head.min_e, head.max_e), eager_weight_norm_init=True)
    model.set_weights(original.get_weights())
    probe = np.random.default_rng(5).normal(0, 1, (1, 48, 48, n_in)).astype(np.float32)
    if not np.array_equal(model(probe).numpy(), original(probe).numpy()):
        raise SystemExit("export_onnx: rebuilt model differs from the checkpoint model")
    return model, n_in


def synthetic_input(knees, size: int = 128) -> np.ndarray:
    """A noisy 4-band field with point sources from 10 to 1e4 e⁻, stretched."""
    rng = np.random.default_rng(0)
    e = rng.normal(0, 3, (size, size, 4)).astype(np.float32)
    yy, xx = np.mgrid[:size, :size]
    for _ in range(20):
        cx, cy, a = rng.uniform(0, size), rng.uniform(0, size), 10 ** rng.uniform(1, 4)
        e += (a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 8.0))[..., None]
    return np.concatenate([np.arcsinh(e / q) for q in knees], -1)[None].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint_dir")
    parser.add_argument("output")
    args = parser.parse_args()

    knees = infer_checkpoint_asinh_knees(args.checkpoint_dir)
    model, n_in = plain_graph_model(args.checkpoint_dir)
    if not knees or len(knees) * 4 != n_in:
        raise SystemExit(f"export_onnx: knees {knees} do not match {n_in} input channels")
    spec = (tf.TensorSpec((1, None, None, n_in), tf.float32, name="lr"),)

    @tf.function(input_signature=spec)
    def forward(x):
        return model(x, training=False)

    proto, _ = tf2onnx.convert.from_function(forward, input_signature=spec, opset=17)
    for dead in [i for i in proto.graph.input if i.name != "lr"]:
        proto.graph.input.remove(dead)
    onnx.checker.check_model(proto)
    onnx.save(proto, args.output)

    x = synthetic_input(knees)
    ref = forward(tf.constant(x)).numpy()
    out = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"]).run(None, {"lr": x})[0]
    rel = float(np.max(np.abs(out - ref) / (np.abs(ref) + 1e-3)))
    print(f"wrote {args.output}: knees {list(knees)}, max relative difference vs TF {rel:.2e}")
    return 0 if rel < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
