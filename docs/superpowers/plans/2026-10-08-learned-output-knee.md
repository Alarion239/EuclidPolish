# Learned Output Knee Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single-image multi-knee member whose output knee is one trainable value per band (the model outputs electrons), then one such member trained on FASRC.

**Architecture:** A new Keras layer `LearnedOutputKnee` (`x = k·sinh(clip(y, ±20))`, `k` bounded to 0.1–10⁴ e⁻ through a sigmoid) is appended to the WDSR graph. Everything downstream already speaks electrons (ensemble, gate, evaluators); the training loss and validation stretch the electron output at the six training knees exactly as option 2 does, so with `k = 10` the objective equals member 196's. Checkpoints carry the head; introspection detects it by the weight's attribute name `output_knee_logit`.

**Tech Stack:** TensorFlow 2.19 / tf_keras, NumPy, pytest; FASRC SLURM through the console's `ensemble_train` step code.

Spec: `docs/superpowers/specs/2026-10-08-learned-output-knee-design.md`.

**Conventions for every task**

- Python: `~/miniforge3/envs/EuclidPolishEnv/bin/python` (written `$PY` below); run from the repo root.
- Test command prefix (written `$T` below):
  `EUCLID_POLISH_DISABLE_AUTO_SSH=1 NUMBA_DISABLE_JIT=1 MPLBACKEND=Agg PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 KMP_DUPLICATE_LIB_OK=TRUE TF_CPP_MIN_LOG_LEVEL=2 $PY -m pytest`
- Commit only the files the task names (`git add <paths>`; the working tree has unrelated user changes — `paper_figures/build_figures.py`, collage files — never stage them). End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- New tests go in `tests/test_learned_output_knee.py` (created in Task 1, extended in later tasks).

---

### Task 1: The `LearnedOutputKnee` layer

**Files:**
- Create: `euclid_polish/training/models/output_knee.py`
- Test: `tests/test_learned_output_knee.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_learned_output_knee.py`:

```python
"""Learned output knee: a single-image multi-knee member whose last layer
learns one asinh output knee per band and outputs electrons."""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
from tf_keras.layers import Input, Lambda
from tf_keras.models import Model as KerasModel

from euclid_polish.training.models.output_knee import (
    KNEE_MAX_E,
    KNEE_MIN_E,
    LearnedOutputKnee,
    knee_logit_for,
    knees_from_logits,
    learned_output_knees,
)

KNEES = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def test_head_starts_at_its_initial_knee_in_every_band():
    head = LearnedOutputKnee(init_knee_e=10.0)
    head.build((None, None, None, 4))
    np.testing.assert_allclose(head.knees().numpy(), [10.0] * 4, rtol=1e-5)


def test_head_maps_asinh_values_to_electrons_with_its_knee():
    head = LearnedOutputKnee(init_knee_e=10.0)
    y = tf.constant(np.linspace(-3.0, 25.0, 32).reshape(1, 2, 4, 4).astype(np.float32))
    expected = 10.0 * np.sinh(np.clip(y.numpy(), -20.0, 20.0))
    np.testing.assert_allclose(head(y).numpy(), expected, rtol=1e-5)


@pytest.mark.parametrize("logit", [-1.0e4, 1.0e4])
def test_head_knee_stays_inside_its_bounds(logit):
    head = LearnedOutputKnee()
    head.build((None, None, None, 4))
    head.output_knee_logit.assign([logit] * 4)
    k = head.knees().numpy()
    assert np.all(k >= KNEE_MIN_E * (1 - 1e-5)) and np.all(k <= KNEE_MAX_E * (1 + 1e-5))


def test_head_knee_gets_a_gradient():
    head = LearnedOutputKnee()
    y = tf.constant(np.full((1, 2, 2, 4), 3.0, np.float32))
    with tf.GradientTape() as tape:
        total = tf.reduce_sum(head(y))
    grad = tape.gradient(total, head.output_knee_logit)
    assert grad is not None and np.all(np.abs(grad.numpy()) > 0)


def test_knee_logit_round_trips_and_rejects_knees_outside_the_bounds():
    for knee in (0.5, 10.0, 3000.0):
        assert knees_from_logits([knee_logit_for(knee)])[0] == pytest.approx(knee, rel=1e-6)
    for knee in (KNEE_MIN_E, KNEE_MAX_E, 0.0, -1.0):
        with pytest.raises(ValueError):
            knee_logit_for(knee)


def test_learned_output_knees_finds_the_head_of_a_model():
    inp = Input(shape=(None, None, 4))
    model = KerasModel(inp, LearnedOutputKnee(init_knee_e=30.0)(inp))
    np.testing.assert_allclose(learned_output_knees(model), [30.0] * 4, rtol=1e-5)
    plain = KerasModel(inp, Lambda(lambda t: t)(inp))
    assert learned_output_knees(plain) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$T tests/test_learned_output_knee.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'euclid_polish.training.models.output_knee'`.

- [ ] **Step 3: Write the layer**

Create `euclid_polish/training/models/output_knee.py`:

```python
"""The learned output knee of a single-image multi-knee member.

A single-image multi-knee member predicts one 4-band image ``y`` in asinh
space; its electrons are ``x = k · sinh(y)``. With a fixed ``k`` (the
``output_knee``, 10 e⁻ for members 195/196) the knee is a hand-picked number.
:class:`LearnedOutputKnee` makes ``k`` one trainable value per band, so the
member learns where its output turns from linear to logarithmic. The layer is
the model's last, so the model outputs electrons.

The knee is bounded to the knee-integrated PSNR range, ``KNEE_MIN_E`` to
``KNEE_MAX_E``, through a sigmoid (no dead gradient at a hard clip), and its
logit is scaled by ``KNEE_LOGIT_SCALE`` so it moves slower than the network's
weights. Checkpoint introspection finds the head by its weight's attribute
name, :data:`KNEE_WEIGHT_NAME`.
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
from tf_keras.initializers import Constant
from tf_keras.layers import Layer

#: Attribute name of the trainable weight: the checkpoint key
#: (``model/layer_with_weights-N/output_knee_logit/…``) introspection looks for.
KNEE_WEIGHT_NAME = "output_knee_logit"
#: Layer name, so a built model's head can be found.
LAYER_NAME = "learned_output_knee"
#: Bounds of the learned knee (electrons): the knee-integrated PSNR range.
KNEE_MIN_E = 0.1
KNEE_MAX_E = 1.0e4
#: Starting knee a loader builds the head with (the restore overwrites it).
DEFAULT_INIT_KNEE_E = 10.0
#: The logit is multiplied by this before the sigmoid, so under Adam the knee
#: moves about ten times slower than an unscaled logit would.
KNEE_LOGIT_SCALE = 0.1
#: Stretched values are clipped to ±this before ``sinh`` (as every member's
#: inference does), so the output stays finite.
SINH_CLIP = 20.0

_LOG_MIN = math.log(KNEE_MIN_E)
_LOG_SPAN = math.log(KNEE_MAX_E) - math.log(KNEE_MIN_E)


def knee_logit_for(knee_e: float) -> float:
    """The weight value at which the head's knee equals ``knee_e`` (e⁻)."""
    knee = float(knee_e)
    if not KNEE_MIN_E < knee < KNEE_MAX_E:
        raise ValueError(f"a learned output knee must start inside "
                         f"({KNEE_MIN_E:g}, {KNEE_MAX_E:g}) e⁻, got {knee_e!r}")
    frac = (math.log(knee) - _LOG_MIN) / _LOG_SPAN
    return math.log(frac / (1.0 - frac)) / KNEE_LOGIT_SCALE


def knees_from_logits(logits) -> np.ndarray:
    """Knees (e⁻) of the head's weight values (NumPy, for checkpoints)."""
    z = KNEE_LOGIT_SCALE * np.asarray(logits, np.float64)
    return np.exp(_LOG_MIN + _LOG_SPAN / (1.0 + np.exp(-z)))


class LearnedOutputKnee(Layer):
    """``x_b = k_b · sinh(clip(y_b, ±SINH_CLIP))`` with one trainable knee
    ``k_b`` per band, bounded to ``(KNEE_MIN_E, KNEE_MAX_E)`` electrons."""

    def __init__(self, init_knee_e: float = DEFAULT_INIT_KNEE_E, **kwargs):
        kwargs.setdefault("name", LAYER_NAME)
        super().__init__(**kwargs)
        self.init_knee_e = float(init_knee_e)
        self._init_logit = knee_logit_for(self.init_knee_e)

    def build(self, input_shape):
        self.output_knee_logit = self.add_weight(
            name=KNEE_WEIGHT_NAME, shape=(int(input_shape[-1]),),
            initializer=Constant(self._init_logit), trainable=True)
        super().build(input_shape)

    def knees(self) -> tf.Tensor:
        """The current knee of every band, electrons, ``(bands,)``."""
        z = KNEE_LOGIT_SCALE * self.output_knee_logit
        return tf.exp(_LOG_MIN + _LOG_SPAN * tf.sigmoid(z))

    def call(self, y):
        k = tf.cast(self.knees(), y.dtype)
        return k * tf.sinh(tf.clip_by_value(y, -SINH_CLIP, SINH_CLIP))

    def get_config(self):
        return {**super().get_config(), "init_knee_e": self.init_knee_e}


def learned_output_knees(model) -> np.ndarray | None:
    """The knees (e⁻) of ``model``'s learned output head, else ``None``."""
    for layer in getattr(model, "layers", ()):
        if isinstance(layer, LearnedOutputKnee):
            return layer.knees().numpy()
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$T tests/test_learned_output_knee.py -q`
Expected: `7 passed` (the bound test is parametrized twice).

- [ ] **Step 5: Commit**

```bash
git add euclid_polish/training/models/output_knee.py tests/test_learned_output_knee.py
git commit -m "Add a learnable per-band output knee layer

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The head in WDSR, and checkpoints that carry it

**Files:**
- Modify: `euclid_polish/training/models/wdsr.py` (signature of `wdsr`, the end of the function, the module docstring)
- Modify: `euclid_polish/training/inference.py` (`infer_checkpoint_num_res_blocks`, new `infer_checkpoint_learned_output_knee`, `load_model_from_checkpoint`)
- Test: `tests/test_learned_output_knee.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_learned_output_knee.py` (and add these imports at the top of the file):

```python
from euclid_polish.training.inference import (
    infer_checkpoint_learned_output_knee,
    infer_checkpoint_nchan_in,
    infer_checkpoint_nchan_out,
    infer_checkpoint_num_res_blocks,
    load_model_from_checkpoint,
)
from euclid_polish.training.models.wdsr import wdsr
```

```python
def _learned_wdsr(blocks: int = 1, knee: float = 10.0):
    return wdsr(scale=2, num_res_blocks=blocks, nchan_in=24, nchan_out=4, input_knees=6,
                learned_output_knee=knee)


def _x24(seed: int):
    return tf.constant(np.random.default_rng(seed).normal(size=(1, 6, 6, 24)).astype(np.float32))


def test_wdsr_learned_head_outputs_electrons_from_the_stretched_image():
    model = _learned_wdsr()
    assert isinstance(model.layers[-1], LearnedOutputKnee)
    pre = KerasModel(model.inputs, model.layers[-2].output)
    x = _x24(1)
    expected = 10.0 * np.sinh(np.clip(pre(x).numpy(), -20.0, 20.0))
    np.testing.assert_allclose(model(x).numpy(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("blocks", [1, 2])
def test_checkpoint_with_a_learned_head_keeps_depth_channels_and_knees(tmp_path, blocks):
    learned = (3.0, 7.0, 20.0, 400.0)
    model = _learned_wdsr(blocks=blocks)
    model.layers[-1].output_knee_logit.assign([knee_logit_for(q) for q in learned])
    d = str(tmp_path / "ckpt")
    tf.train.Checkpoint(model=model).save(d + "/ckpt")
    assert infer_checkpoint_num_res_blocks(d) == blocks
    assert infer_checkpoint_nchan_in(d) == 24
    assert infer_checkpoint_nchan_out(d, scale=2, nchan_in=24) == 4
    assert infer_checkpoint_learned_output_knee(d) == pytest.approx(learned, rel=1e-5)
    loaded = load_model_from_checkpoint(d, scale=2, num_res_blocks=32)
    np.testing.assert_allclose(learned_output_knees(loaded), learned, rtol=1e-5)
    x = _x24(2)
    np.testing.assert_allclose(loaded(x).numpy(), model(x).numpy(), rtol=1e-5, atol=1e-5)


def test_checkpoint_without_the_head_reports_no_learned_knee(tmp_path):
    model = wdsr(scale=2, num_res_blocks=1, nchan_in=24, nchan_out=4, input_knees=6)
    d = str(tmp_path / "ckpt")
    tf.train.Checkpoint(model=model).save(d + "/ckpt")
    assert infer_checkpoint_learned_output_knee(d) is None
    assert infer_checkpoint_num_res_blocks(d) == 1
    assert learned_output_knees(load_model_from_checkpoint(d, scale=2, num_res_blocks=1)) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$T tests/test_learned_output_knee.py -q`
Expected: collection error, `ImportError: cannot import name 'infer_checkpoint_learned_output_knee'`.

- [ ] **Step 3: Add the head to `wdsr`**

In `euclid_polish/training/models/wdsr.py`:

1. After the existing `from euclid_polish.training.models.common import ICNR, pixel_shuffle` import add:

```python
from euclid_polish.training.models.output_knee import LearnedOutputKnee
```

2. Append this paragraph to the end of the module docstring (before its closing `"""`):

```text
Exception: a member with a learned output knee (``learned_output_knee``)
ends in :class:`~euclid_polish.training.models.output_knee.LearnedOutputKnee`,
which turns the stretched output into electrons with one trainable knee per
band, so that model outputs electrons.
```

3. Change the signature line

```python
         per_band_skip=None, icnr=False, input_knees=1):
```

to

```python
         per_band_skip=None, icnr=False, input_knees=1,
         learned_output_knee=None):
```

4. In the docstring's parameter list, after the `input_knees` entry, add:

```text
    learned_output_knee : starting knee (e⁻) of a learned output head: the
                   model then ends in ``LearnedOutputKnee`` and outputs
                   electrons, one trainable knee per output band. ``None``
                   (default) keeps the stretched output.
```

5. Replace the last two lines of the function

```python
    x = Add()([m, s])
    return Model(x_in, x, name="wdsr")
```

with

```python
    x = Add()([m, s])
    if learned_output_knee is not None:
        # A learned output knee: one trainable asinh knee per band turns the
        # stretched output into electrons, so the model outputs electrons.
        x = LearnedOutputKnee(init_knee_e=learned_output_knee)(x)
    return Model(x_in, x, name="wdsr")
```

- [ ] **Step 4: Teach checkpoint introspection and loading about the head**

In `euclid_polish/training/inference.py`:

1. Add to the imports (after `from euclid_polish.training.models.common import resolve_single`):

```python
from euclid_polish.training.models.output_knee import (
    DEFAULT_INIT_KNEE_E,
    KNEE_WEIGHT_NAME,
    knees_from_logits,
)
```

2. In `infer_checkpoint_num_res_blocks`, replace the loop and the depth computation

```python
    layers: set[int] = set()
    skip_layers: set[int] = set()
    for key, shp in shapes.items():
        m = _MODEL_LAYER_KEY.match(key)
        if m is None:
            continue
        idx = int(m.group(1))
        layers.add(idx)
        if (len(shp) == 4 and shp[0] == skip_kernel_size
                and shp[1] == skip_kernel_size):
            skip_layers.add(idx)
    if not layers or not skip_layers:
        return None
    trunk = len(layers) - 2 - len(skip_layers)
```

with

```python
    layers: set[int] = set()
    skip_layers: set[int] = set()
    head_layers: set[int] = set()
    for key, shp in shapes.items():
        m = _MODEL_LAYER_KEY.match(key)
        if m is None:
            continue
        idx = int(m.group(1))
        layers.add(idx)
        if (len(shp) == 4 and shp[0] == skip_kernel_size
                and shp[1] == skip_kernel_size):
            skip_layers.add(idx)
        if f"/{KNEE_WEIGHT_NAME}/" in key:
            head_layers.add(idx)        # the learned output knee: not a conv
    if not layers or not skip_layers:
        return None
    trunk = len(layers - head_layers) - 2 - len(skip_layers)
```

and add `(and, when present, the learned output knee, which is not counted)` after `+ the skip conv(s)` in that function's docstring.

3. Directly after `infer_checkpoint_num_res_blocks`, add:

```python
def infer_checkpoint_learned_output_knee(checkpoint_dir: str) -> tuple[float, ...] | None:
    """The per-band learned output knees (e⁻) a checkpoint's model carries,
    or ``None`` for a model without the learned head. Like depth, this is
    read from the checkpoint itself: the head changes the layer graph and the
    meaning of the output (electrons instead of asinh)."""
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is None:
        return None
    suffix = f"/{KNEE_WEIGHT_NAME}/.ATTRIBUTES/VARIABLE_VALUE"
    try:
        reader = tf.train.load_checkpoint(latest)
        keys = [key for key in reader.get_variable_to_shape_map()
                if key.startswith("model/") and key.endswith(suffix)]
        if not keys:
            return None
        return tuple(float(q) for q in knees_from_logits(reader.get_tensor(keys[0])))
    except Exception:    # pragma: no cover — unreadable ckpt → no head
        return None
```

4. In `load_model_from_checkpoint`, replace

```python
    model = wdsr(
        scale=scale, num_res_blocks=num_res_blocks,
        nchan_in=nchan_in, nchan_out=nchan_out, input_knees=input_knees,
    )
```

with

```python
    # A learned output head is part of the layer graph: build it when the
    # checkpoint carries one (the restore below sets its learned knees).
    learned = infer_checkpoint_learned_output_knee(checkpoint_dir)
    model = wdsr(
        scale=scale, num_res_blocks=num_res_blocks,
        nchan_in=nchan_in, nchan_out=nchan_out, input_knees=input_knees,
        learned_output_knee=(DEFAULT_INIT_KNEE_E if learned is not None else None),
    )
```

and replace the final print

```python
    print(f"Model restored from checkpoint at {latest} "
          f"(nchan_in={nchan_in}, nchan_out={nchan_out}).")
```

with

```python
    head = ("" if learned is None else
            ", learned output knees " + "/".join(f"{q:.3g}" for q in learned) + " e⁻")
    print(f"Model restored from checkpoint at {latest} "
          f"(nchan_in={nchan_in}, nchan_out={nchan_out}{head}).")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$T tests/test_learned_output_knee.py tests/test_multi_knee.py tests/test_inference_nchan_in.py tests/test_inference_shapes.py -q`
Expected: all pass (the existing multi-knee and introspection tests still pass).

- [ ] **Step 6: Commit**

```bash
git add euclid_polish/training/models/wdsr.py euclid_polish/training/inference.py tests/test_learned_output_knee.py
git commit -m "Build and restore WDSR members with a learned output knee

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Loss, validation and logging for an electron output

**Files:**
- Modify: `euclid_polish/training/losses.py` (import, new `knee_stretched_loss`)
- Modify: `euclid_polish/training/models/common.py` (import, `evaluate`, `_evaluate_multi_knee`)
- Modify: `euclid_polish/training/trainer.py` (`Trainer.__init__`, `Trainer.evaluate`, `Trainer._validate`)
- Test: `tests/test_learned_output_knee.py`

- [ ] **Step 1: Write the failing tests**

Add imports to the test file:

```python
from euclid_polish.training.augmentation import asinh_stretch_multi_knee, stretch_pair
from euclid_polish.training.losses import (
    build_loss,
    channel_balanced_loss,
    knee_expanded_loss,
    knee_stretched_loss,
)
from euclid_polish.training.models.common import evaluate
from euclid_polish.training.trainer import Trainer
```

Append:

```python
def test_learned_knee_loss_and_validation_equal_option_2_at_the_initial_knee():
    rng = np.random.default_rng(3)
    hr_e = tf.constant(rng.uniform(0.0, 3000.0, (1, 8, 8, 4)).astype(np.float32))
    target = asinh_stretch_multi_knee(hr_e, KNEES)
    y = tf.asinh(hr_e / 10.0) + 0.01          # an option-2 output, slightly off
    x = 10.0 * tf.sinh(y)                     # the same image in electrons
    for base in (build_loss("l2"), channel_balanced_loss("l2")):
        np.testing.assert_allclose(float(knee_stretched_loss(base, KNEES)(x, target)),
                                   float(knee_expanded_loss(base, 10.0, KNEES)(y, target)),
                                   rtol=1e-4)
    lr = tf.zeros((1, 4, 4, 24))
    learned = evaluate(lambda _lr: x, [(lr, target)], knees=KNEES, output_electrons=True)
    fixed = evaluate(lambda _lr: y, [(lr, target)], knees=KNEES, output_knee=10.0)
    for key in ("psnr_stretched", "psnr_raw", "mae_stretched"):
        np.testing.assert_allclose(float(learned[key]), float(fixed[key]), rtol=1e-4)


def test_trainer_steps_a_learned_knee_member_and_logs_its_knees(tmp_path, capsys):
    rng = np.random.default_rng(7)
    lr_e = tf.constant(rng.uniform(0, 200, (2, 8, 8, 4)).astype(np.float32))
    hr_e = tf.constant(rng.uniform(0, 200, (2, 16, 16, 4)).astype(np.float32))
    lr, hr = stretch_pair(lr_e, hr_e, knees=KNEES)
    model = _learned_wdsr()
    loss = knee_stretched_loss(channel_balanced_loss("l2"), KNEES)
    trainer = Trainer(model, loss=loss, learning_rate=1e-2, checkpoint_dir=str(tmp_path),
                      knees=KNEES, output_electrons=True)
    before = learned_output_knees(model).copy()
    value, gnorm = trainer.train_step(lr, hr)
    assert np.isfinite(float(value)) and np.isfinite(float(gnorm))
    assert not np.allclose(learned_output_knees(model), before)     # the knee trains
    assert np.isfinite(trainer._validate(tf.data.Dataset.from_tensors((lr, hr)), 1)["psnr_str"])
    assert "learned output knee" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$T tests/test_learned_output_knee.py -q`
Expected: collection error, `ImportError: cannot import name 'knee_stretched_loss'`.

- [ ] **Step 3: Add the loss**

In `euclid_polish/training/losses.py`, change

```python
from euclid_polish.training.augmentation import expand_to_knees
```

to

```python
from euclid_polish.training.augmentation import asinh_stretch_multi_knee, expand_to_knees
```

and append at the end of the file:

```python
def knee_stretched_loss(loss, knees):
    """Loss of a member that outputs ELECTRONS (a learned output knee): its
    output is stretched at every knee and compared, channel for channel, with
    the knee-major multi-knee target by ``loss`` (plain or channel-balanced);
    signature ``loss(sr, hr)``. With the head's knee at ``k`` this is
    :func:`knee_expanded_loss` at output knee ``k``."""
    def _loss(sr, hr):
        return loss(asinh_stretch_multi_knee(sr, knees), hr)
    return _loss
```

- [ ] **Step 4: Score an electron output in validation**

In `euclid_polish/training/models/common.py`:

1. Change `from euclid_polish.training.augmentation import expand_to_knees` to

```python
from euclid_polish.training.augmentation import asinh_stretch_multi_knee, expand_to_knees
```

2. Change the `evaluate` signature

```python
def evaluate(model, dataset, knees: Sequence[float] | None = None,
             output_knee: float | None = None, knee: float | None = None):
```

to

```python
def evaluate(model, dataset, knees: Sequence[float] | None = None,
             output_knee: float | None = None, knee: float | None = None,
             output_electrons: bool = False):
```

add to its docstring, after the sentence that ends `scored at every knee.`:

```text
    ``output_electrons`` marks one whose output is already electrons (a
    learned output knee); it is stretched at every knee first.
```

and change

```python
    if knees:
        return _evaluate_multi_knee(model, dataset, knees, output_knee)
```

to

```python
    if knees:
        return _evaluate_multi_knee(model, dataset, knees, output_knee,
                                    output_electrons=output_electrons)
```

3. Change the `_evaluate_multi_knee` signature to

```python
def _evaluate_multi_knee(model, dataset, knees: Sequence[float],
                         output_knee: float | None = None,
                         output_electrons: bool = False) -> dict:
```

append to its docstring `A member whose output is electrons (output_electrons) is stretched at every knee first.`, and replace

```python
        sr = model(lr)
        if output_knee is not None:
            sr = expand_to_knees(sr, output_knee, knees)
```

with

```python
        sr = model(lr)
        if output_electrons:
            sr = asinh_stretch_multi_knee(sr, knees)
        elif output_knee is not None:
            sr = expand_to_knees(sr, output_knee, knees)
```

- [ ] **Step 5: Pass it through the trainer and log the knees**

In `euclid_polish/training/trainer.py`:

1. Add the import (next to the other `euclid_polish.training.models` imports):

```python
from euclid_polish.training.models.output_knee import learned_output_knees
```

2. In `Trainer.__init__`'s signature, after `output_knee: float | None = None,` add `output_electrons: bool = False,`; after the line `self._output_knee = float(output_knee) if output_knee is not None else None` add:

```python
        # A member whose output is electrons (a learned output knee): its
        # output is stretched at every knee for validation.
        self._output_electrons = bool(output_electrons)
```

3. In `Trainer.evaluate`, replace

```python
            return evaluate(self.checkpoint.model, dataset, knees=self._knees,
                            output_knee=self._output_knee)
```

with

```python
            return evaluate(self.checkpoint.model, dataset, knees=self._knees,
                            output_knee=self._output_knee,
                            output_electrons=self._output_electrons)
```

4. In `Trainer._validate`, directly after the `if knee_vals is not None and self._knees:` block that prints `per-knee PSNR`, add:

```python
        learned = learned_output_knees(self.checkpoint.model)
        if learned is not None:
            tqdm.write("  learned output knee (e⁻): " + " | ".join(
                f"{band} {float(q):.3g}" for band, q in
                zip(Config.HR_TARGET_BAND_NAMES, learned, strict=False)))
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `$T tests/test_learned_output_knee.py tests/test_multi_knee.py tests/test_trainer_psnr_raw_knee.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add euclid_polish/training/losses.py euclid_polish/training/models/common.py euclid_polish/training/trainer.py tests/test_learned_output_knee.py
git commit -m "Train and validate members that output electrons through a learned knee

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Inference and the `Model` wrapper

**Files:**
- Modify: `euclid_polish/training/inference.py` (`reconstruct`)
- Modify: `euclid_polish/model.py` (imports, `Model.__init__`, new `Model._member_loss`, `Model.train`, `Model._knee_kw`, `Model.upsample_heads`)
- Test: `tests/test_learned_output_knee.py`

- [ ] **Step 1: Write the failing tests**

Add imports to the test file:

```python
import json

from tf_keras.layers import UpSampling2D

from euclid_polish.model import Model
from euclid_polish.training.inference import reconstruct
```

Append:

```python
def test_reconstruct_returns_a_learned_knee_members_electrons_unchanged():
    inp = Input(shape=(None, None, 24))
    electrons = Lambda(lambda t: 1.0e4 * tf.sinh(t[..., 20:24]))(inp)   # the 10⁴ e⁻ block
    model = KerasModel(inp, UpSampling2D(size=2, interpolation="nearest")(electrons))
    x = np.random.default_rng(6).uniform(1.0, 500.0, (6, 6, 4)).astype(np.float32)
    _lr, sr = reconstruct(model, x, knees=KNEES, output_electrons=True)
    np.testing.assert_allclose(sr, np.kron(x, np.ones((2, 2, 1), np.float32)), rtol=1e-4)


def test_fresh_learned_knee_member_builds_the_head_and_infers_in_electrons(tmp_path):
    m = Model(str(tmp_path / "m"), scale=2, num_res_blocks=1, asinh_knees=KNEES,
              output_knee=10.0, learn_output_knee=True)
    assert m._tf_model.inputs[0].shape[-1] == 24 and m._tf_model.outputs[0].shape[-1] == 4
    np.testing.assert_allclose(learned_output_knees(m._tf_model), [10.0] * 4, rtol=1e-5)
    assert m._learn_output_knee and m._output_knee is None
    assert m._knee_kw() == {"knees": KNEES, "output_electrons": True}
    sr = m.upsample_array(np.random.default_rng(8).uniform(0, 50, (6, 6, 4)).astype(np.float32))
    assert sr.shape == (12, 12, 4) and np.all(np.isfinite(sr))
    with pytest.raises(ValueError):
        m.upsample_heads(np.zeros((4, 4, 4), np.float32))
    with pytest.raises(ValueError):          # no starting knee
        Model(str(tmp_path / "x"), num_res_blocks=1, asinh_knees=KNEES, learn_output_knee=True)


def test_learned_knee_member_resumes_with_its_head(tmp_path):
    d = tmp_path / "member"
    m = Model(str(d), scale=2, num_res_blocks=1, asinh_knees=KNEES, output_knee=10.0,
              learn_output_knee=True)
    m._tf_model.layers[-1].output_knee_logit.assign(
        [knee_logit_for(q) for q in (2.0, 5.0, 8.0, 9.0)])
    tf.train.Checkpoint(model=m._tf_model).save(str(d / "ckpt"))
    (d / "origin.json").write_text(json.dumps({
        "asinh_knees": list(KNEES), "knee_loss": "balanced",
        "learned_output_knee": {"init_e": 10.0, "min_e": 0.1, "max_e": 10000.0}}))
    r = Model(str(d), scale=2, num_res_blocks=32)
    assert r._learn_output_knee and r._output_knee is None and r._num_res_blocks == 1
    assert r._knee_kw() == {"knees": KNEES, "output_electrons": True}
    np.testing.assert_allclose(learned_output_knees(r._tf_model), (2.0, 5.0, 8.0, 9.0), rtol=1e-5)
    x = np.random.default_rng(9).uniform(0, 50, (6, 6, 4)).astype(np.float32)
    np.testing.assert_allclose(r.upsample_array(x), m.upsample_array(x), rtol=1e-5, atol=1e-4)


def test_a_fork_of_a_learned_knee_member_keeps_the_head(tmp_path):
    src = tmp_path / "src"
    m = Model(str(src), scale=2, num_res_blocks=1, asinh_knees=KNEES, output_knee=10.0,
              learn_output_knee=True)
    m._tf_model.layers[-1].output_knee_logit.assign(
        [knee_logit_for(q) for q in (4.0, 4.0, 6.0, 6.0)])
    tf.train.Checkpoint(model=m._tf_model).save(str(src / "ckpt"))
    (src / "origin.json").write_text(json.dumps({
        "asinh_knees": list(KNEES), "knee_loss": "balanced",
        "learned_output_knee": {"init_e": 10.0, "min_e": 0.1, "max_e": 10000.0}}))
    fork = Model(str(tmp_path / "fork"), scale=2, num_res_blocks=32, init_weights_from=str(src))
    assert fork._learn_output_knee and fork._output_knee is None
    np.testing.assert_allclose(learned_output_knees(fork._tf_model), (4.0, 4.0, 6.0, 6.0), rtol=1e-5)


def test_learned_knee_member_trains_under_the_stretched_loss(tmp_path):
    m = Model(str(tmp_path / "m"), scale=2, num_res_blocks=1, asinh_knees=KNEES,
              output_knee=10.0, learn_output_knee=True)
    hr_e = tf.constant(np.random.default_rng(10).uniform(0, 3000, (1, 8, 8, 4)).astype(np.float32))
    target = asinh_stretch_multi_knee(hr_e, KNEES)
    expected = knee_stretched_loss(channel_balanced_loss("l2"), KNEES)(hr_e * 1.1, target)
    got = m._member_loss("l2", "balanced")(hr_e * 1.1, target)
    np.testing.assert_allclose(float(got), float(expected), rtol=1e-6)
    fixed = Model(str(tmp_path / "f"), scale=2, num_res_blocks=1, asinh_knees=KNEES,
                  output_knee=10.0)
    y = tf.asinh(hr_e * 1.1 / 10.0)
    np.testing.assert_allclose(float(fixed._member_loss("l2", "balanced")(y, target)),
                               float(expected), rtol=1e-4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$T tests/test_learned_output_knee.py -q`
Expected: the five new tests FAIL (`TypeError: reconstruct() got an unexpected keyword argument 'output_electrons'`, `TypeError: Model.__init__() got an unexpected keyword argument 'learn_output_knee'`).

- [ ] **Step 3: `reconstruct` returns an electron output as is**

In `euclid_polish/training/inference.py`, change `reconstruct`'s signature

```python
    head_knee: float | None = None,
    output_knee: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
```

to

```python
    head_knee: float | None = None,
    output_knee: float | None = None,
    output_electrons: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
```

document it after the `output_knee` entry of the docstring:

```text
    output_electrons : bool, optional
        A multi-knee member with a learned output knee: the input is
        stretched at every knee and the model's output, already electrons,
        is returned unchanged.
```

and insert as the first branch of the body (before `if knees and output_knee is not None:`):

```python
    if knees and output_electrons:
        lr_data, lr_for_model = _model_input(model, lr_input, len(knees))
        stretched = asinh_stretch_multi_knee(tf.constant(lr_for_model), knees)
        # The model's last layer (the learned output knee) returns electrons.
        sr_data = resolve_single(model, stretched).numpy().astype(np.float32)
        lr_display = lr_data[..., 0] if lr_data.ndim == 3 else lr_data
        return lr_display, sr_data
```

- [ ] **Step 4: `Model` builds, resumes, trains and infers the learned head**

In `euclid_polish/model.py`:

1. Add next to the other `euclid_polish.training.inference` imports:

```python
from euclid_polish.training.inference import (
    infer_checkpoint_learned_output_knee as _infer_learned_output_knee,
)
```

and change the losses import to

```python
from euclid_polish.training.losses import (
    KNEE_LOSS_MODES,
    build_loss,
    channel_balanced_loss,
    knee_expanded_loss,
    knee_stretched_loss,
)
```

2. In `Model.__init__`'s signature, after `output_knee: float | None = None,` add `learn_output_knee: bool = False,`. After the block that sets `self._output_knee` (ending `float(output_knee) if output_knee is not None else None)`), add:

```python
        # Learned output knee: one trainable knee per band replaces the fixed
        # ``output_knee``, which is only its starting value; the model then
        # outputs electrons. Architecture-bound, so a resume/fork reads it
        # from the checkpoint (below) and only a fresh build uses the flag.
        self._learn_output_knee = bool(learn_output_knee)
```

3. In the resume branch, after `self._output_knee = _infer_output_knee(checkpoint_dir)` add:

```python
            self._learn_output_knee = (
                _infer_learned_output_knee(checkpoint_dir) is not None)
            if self._learn_output_knee:
                self._output_knee = None
```

4. In the fork branch, after `self._output_knee = _infer_output_knee(init_weights_from)` add:

```python
            self._learn_output_knee = (
                _infer_learned_output_knee(init_weights_from) is not None)
            if self._learn_output_knee:
                self._output_knee = None
```

5. In the fresh-build branch, replace

```python
            n_knees = len(self._asinh_knees) if self._asinh_knees else 1
            single = self._output_knee is not None
            self._tf_model = _wdsr_build(
                scale=scale, num_res_blocks=num_res_blocks,
                nchan_in=Config.NUM_LR_CHANNELS * n_knees,
                nchan_out=Config.NUM_HR_CHANNELS * (1 if single else n_knees),
                input_knees=n_knees if single else 1,
                icnr=self._icnr)
            self.id = None
```

with

```python
            if self._learn_output_knee and (not self._asinh_knees
                                            or self._output_knee is None):
                raise ValueError("learn_output_knee needs asinh_knees and "
                                 "output_knee (its starting value)")
            n_knees = len(self._asinh_knees) if self._asinh_knees else 1
            single = self._output_knee is not None
            self._tf_model = _wdsr_build(
                scale=scale, num_res_blocks=num_res_blocks,
                nchan_in=Config.NUM_LR_CHANNELS * n_knees,
                nchan_out=Config.NUM_HR_CHANNELS * (1 if single else n_knees),
                input_knees=n_knees if single else 1,
                icnr=self._icnr,
                learned_output_knee=(self._output_knee
                                     if self._learn_output_knee else None))
            if self._learn_output_knee:
                self._output_knee = None     # the knee is learned now
            self.id = None
```

6. Replace the loss block in `Model.train`

```python
        # A multi-knee member's loss: ``plain`` = the loss over all channels
        # at once; ``balanced`` = every channel (band x knee) weighted equally.
        if knee_loss not in KNEE_LOSS_MODES:
            raise ValueError(f"knee_loss must be one of {KNEE_LOSS_MODES}, got {knee_loss!r}")
        loss = (channel_balanced_loss(loss_norm)
                if self._asinh_knees and knee_loss == "balanced"
                else build_loss(loss_norm))
        if self._output_knee is not None:
            # One output image, re-stretched at every knee against the target.
            loss = knee_expanded_loss(loss, self._output_knee, self._asinh_knees)
        trainer = Trainer(self._tf_model, learning_rate=lr_schedule,
                          checkpoint_dir=self._checkpoint_dir,
                          loss=loss, knees=self._asinh_knees,
                          output_knee=self._output_knee,
```

with

```python
        loss = self._member_loss(loss_norm, knee_loss)
        trainer = Trainer(self._tf_model, learning_rate=lr_schedule,
                          checkpoint_dir=self._checkpoint_dir,
                          loss=loss, knees=self._asinh_knees,
                          output_knee=self._output_knee,
                          output_electrons=self._learn_output_knee,
```

and add this method directly before `def train(`:

```python
    def _member_loss(self, loss_norm: str, knee_loss: str = "plain"):
        """The reconstruction loss this member trains under. A multi-knee
        member's ``knee_loss``: ``plain`` = the loss over all channels at
        once; ``balanced`` = every channel (band x knee) weighted equally. A
        single-image member's one output is re-stretched at every knee: from
        its fixed output knee, or (learned knee) from electrons."""
        if knee_loss not in KNEE_LOSS_MODES:
            raise ValueError(f"knee_loss must be one of {KNEE_LOSS_MODES}, got {knee_loss!r}")
        loss = (channel_balanced_loss(loss_norm)
                if self._asinh_knees and knee_loss == "balanced"
                else build_loss(loss_norm))
        if self._learn_output_knee:
            return knee_stretched_loss(loss, self._asinh_knees)
        if self._output_knee is not None:
            return knee_expanded_loss(loss, self._output_knee, self._asinh_knees)
        return loss
```

7. In `Model._knee_kw`, replace

```python
        knees = getattr(self, "_asinh_knees", None)
        output_knee = getattr(self, "_output_knee", None)
```

with

```python
        knees = getattr(self, "_asinh_knees", None)
        if knees and getattr(self, "_learn_output_knee", False):
            return {"knees": knees, "output_electrons": True}
        output_knee = getattr(self, "_output_knee", None)
```

8. In `Model.upsample_heads`, replace the guard

```python
        if not getattr(self, "_asinh_knees", None) or getattr(self, "_output_knee", None) is not None:
```

with

```python
        if (not getattr(self, "_asinh_knees", None)
                or getattr(self, "_output_knee", None) is not None
                or getattr(self, "_learn_output_knee", False)):
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$T tests/test_learned_output_knee.py tests/test_multi_knee.py tests/test_model.py tests/test_model_fork_init.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add euclid_polish/training/inference.py euclid_polish/model.py tests/test_learned_output_knee.py
git commit -m "Infer, resume and train members with a learned output knee

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Member spec, `origin.json` and the CLI

**Files:**
- Modify: `euclid_polish/ensemble.py` (`MemberTrainSpec`, `EnsembleModel.train_members`)
- Modify: `scripts/train_ensemble.py` (imports, `--learn-output-knee`, member-spec keys, `_diversity_kwargs`, the knobs printout)
- Test: `tests/test_learned_output_knee.py`

- [ ] **Step 1: Write the failing tests**

Add imports to the test file:

```python
from euclid_polish import ensemble as ens_mod
from euclid_polish.ensemble import EnsembleModel, MemberTrainSpec
from euclid_polish.training.inference import infer_checkpoint_output_knee
from scripts.train_ensemble import build_specs, parse_args
```

Append:

```python
class _LearnedKneeModel:
    """Model stand-in for train_members that resolves the knobs like Model."""

    def __init__(self, checkpoint_dir, *, scale=2, num_res_blocks=32, seed=None,
                 init_weights_from=None, icnr=False, asinh_knee=None, asinh_knees=None,
                 output_knee=None, learn_output_knee=False):
        self._num_res_blocks = num_res_blocks
        self._asinh_knee = asinh_knee
        self._asinh_knees = tuple(asinh_knees) if asinh_knees else None
        self._learn_output_knee = bool(learn_output_knee)
        self._output_knee = None if learn_output_knee else output_knee
        self.trained: dict = {}

    def train(self, lr, hr, **kwargs):
        self.trained = kwargs


def test_train_members_records_a_learned_knee_without_an_output_knee(tmp_path, monkeypatch):
    monkeypatch.setattr(ens_mod, "Model", _LearnedKneeModel)
    base = tmp_path / "ensemble"
    spec = MemberTrainSpec(name="member_206", seed=1, target_steps=10, run_steps=10,
                           loss_norm="l2", asinh_knees=KNEES, knee_loss="balanced",
                           output_knee=10.0, learn_output_knee=True)
    EnsembleModel(str(base), _models=[]).train_members("lr", "hr", [spec])
    origin = json.loads((base / "member_206" / "origin.json").read_text())
    assert origin["learned_output_knee"] == {"init_e": 10.0, "min_e": 0.1, "max_e": 10000.0}
    assert "output_knee" not in origin
    assert infer_checkpoint_output_knee(str(base / "member_206")) is None


def test_member_spec_makes_a_learned_knee_member(tmp_path):
    member = {"asinh_knees": list(KNEES), "output_knee": 10, "knee_loss": "balanced",
              "learn_output_knee": True}
    args = parse_args(["--count", "1", "--steps", "10", "--member-spec", json.dumps([member])])
    spec = build_specs(args, str(tmp_path / "ens"))[0]
    assert spec.learn_output_knee and spec.output_knee == 10.0
    run_wide = parse_args(["--count", "1", "--steps", "10", "--asinh-knees", "0.1,1,10",
                           "--output-knee", "10", "--learn-output-knee"])
    assert build_specs(run_wide, str(tmp_path / "ens2"))[0].learn_output_knee
    plain = parse_args(["--count", "1", "--steps", "10"])
    assert not build_specs(plain, str(tmp_path / "ens3"))[0].learn_output_knee


@pytest.mark.parametrize("member", [
    {"asinh_knees": list(KNEES), "learn_output_knee": True},                      # no starting knee
    {"output_knee": 10, "learn_output_knee": True},                               # not multi-knee
    {"asinh_knees": list(KNEES), "output_knee": 1e5, "learn_output_knee": True},  # outside the bounds
    {"asinh_knees": list(KNEES), "output_knee": 10, "learn_output_knee": 1},      # not a bool
])
def test_member_spec_rejects_bad_learned_knee_settings(member, tmp_path):
    args = parse_args(["--count", "1", "--steps", "10", "--member-spec", json.dumps([member])])
    with pytest.raises(SystemExit):
        build_specs(args, str(tmp_path / "ens"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$T tests/test_learned_output_knee.py -q`
Expected: the new tests FAIL (`TypeError: MemberTrainSpec.__init__() got an unexpected keyword argument 'learn_output_knee'`; the CLI ones exit on the unknown key / flag).

- [ ] **Step 3: The spec field and `origin.json`**

In `euclid_polish/ensemble.py`:

1. Add the import (next to the other `euclid_polish.training` imports):

```python
from euclid_polish.training.models.output_knee import KNEE_MAX_E, KNEE_MIN_E
```

2. In `MemberTrainSpec`, after the `output_knee` field add:

```python
    #: Learn the single-image member's output knee, one value per band
    #: (``output_knee`` is its starting value; the model outputs electrons).
    #: A NEW-member (add) knob; continue/fork read it from the checkpoint.
    learn_output_knee: bool = False
```

3. In `train_members`, after

```python
            if spec.output_knee is not None:
                model_kwargs["output_knee"] = spec.output_knee
```

add

```python
            if spec.learn_output_knee:
                model_kwargs["learn_output_knee"] = True
```

and inside the `if knees:` block that writes the multi-knee fields, after the `output_knee` lines, add:

```python
                    if getattr(m, "_learn_output_knee", False):
                        # No ``output_knee`` key: code unaware of the learned
                        # head would apply k·sinh to electrons; without the
                        # key it fails loudly instead.
                        origin["learned_output_knee"] = {
                            "init_e": (float(spec.output_knee)
                                       if spec.output_knee is not None else None),
                            "min_e": KNEE_MIN_E, "max_e": KNEE_MAX_E}
```

- [ ] **Step 4: The CLI**

In `scripts/train_ensemble.py`:

1. Add the import next to the other `euclid_polish` imports:

```python
from euclid_polish.training.models.output_knee import KNEE_MAX_E, KNEE_MIN_E
```

2. After the `--output-knee` argument add:

```python
    p.add_argument("--learn-output-knee", action="store_true",
                   help="With --asinh-knees and --output-knee: learn the "
                        "output knee during training, one value per band, "
                        "starting at --output-knee (inside 0.1–10⁴ e⁻); the "
                        "member then outputs electrons. ADD members only; "
                        "continue/fork read it from the checkpoint. Member-"
                        "spec key: learn_output_knee (true/false).")
```

3. In `_member_overrides`, add `"learn_output_knee"` to the `allowed` set (after `"output_knee"`).

4. In `_diversity_kwargs`, after the `if output_knee is not None:` validation block add:

```python
    learn = over.get("learn_output_knee", args.learn_output_knee)
    if not isinstance(learn, bool):
        print(f"✗ learn_output_knee must be true or false, got {learn!r}")
        raise SystemExit(2)
    if learn:
        if output_knee is None:
            print("✗ learn_output_knee needs asinh_knees and output_knee "
                  "(its starting value)")
            raise SystemExit(2)
        if not KNEE_MIN_E < output_knee < KNEE_MAX_E:
            print(f"✗ a learned output knee must start inside "
                  f"({KNEE_MIN_E:g}, {KNEE_MAX_E:g}) e⁻, got {output_knee:g}")
            raise SystemExit(2)
```

and add `"learn_output_knee": learn,` to the returned dict (after `"output_knee": output_knee,`).

5. In the knobs printout, replace

```python
            if s.output_knee is not None:
                knobs += f" output_knee={s.output_knee:g}e"
```

with

```python
            if s.output_knee is not None:
                knobs += f" output_knee={s.output_knee:g}e"
                if s.learn_output_knee:
                    knobs += " (learned)"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$T tests/test_learned_output_knee.py tests/test_multi_knee.py tests/test_train_ensemble_specs.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add euclid_polish/ensemble.py scripts/train_ensemble.py tests/test_learned_output_knee.py
git commit -m "Request a learned output knee per member and record it in origin.json

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: README, lint and the full suite

**Files:**
- Modify: `README.md` (§8.2 multi-knee bullets)

- [ ] **Step 1: Document the knob**

In `README.md` §8.2, after the bullet starting `  - *Option 2* (\`--output-knee 10\`)`, add:

```markdown
  - *Learned output knee* (`--learn-output-knee`, with option 2): the output knee becomes one
    trainable value per band, starting at `--output-knee` and bounded to 0.1–10⁴ e⁻, and the
    member outputs electrons. The knees are printed at every validation and live in the
    checkpoint; `origin.json` records `learned_output_knee` instead of `output_knee`.
```

- [ ] **Step 2: Lint**

Run: `~/miniforge3/envs/EuclidPolishEnv/bin/ruff check .`
Expected: `All checks passed!` (fix any finding in the files this plan touched).

- [ ] **Step 3: Full suite**

Run: `$T -q -x`
Expected: all pass (~4,240 tests; the few data-dependent skips as before).

- [ ] **Step 4: Commit and push**

```bash
git add README.md
git commit -m "Document the learned output knee

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 7: Local end-to-end smoke (CPU, record mode, 20 steps)

Exercises spec → `train_members` → `Model` → `Trainer` → validation → checkpoint → `origin.json` → reload → inference on real data, without spending FASRC time. Nothing is committed.

- [ ] **Step 1: Stage stand-in records**

```bash
S=/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/ac23cec7-51a0-4b8d-9ff8-742dabdefc05/scratchpad/learned_knee_smoke
R=$PWD/data/_fasrc_cache/n/netscratch/lconnor_lab/Lab/abelotserkovtsev/EuclidPolish/data/images/records_v2
rm -rf "$S" && mkdir -p "$S/records"
for k in dirty clean hr; do
  ln -s "$R/${k}_validate.tfrecord" "$S/records/${k}_validate.tfrecord"
  ln -s "$R/${k}_validate.tfrecord" "$S/records/${k}_train.tfrecord"
done
```

- [ ] **Step 2: Train 20 steps through `train_members`**

```bash
cd /Users/alarion239/Desktop/EuclidPolish && S=$S KMP_DUPLICATE_LIB_OK=TRUE TF_CPP_MIN_LOG_LEVEL=2 EUCLID_POLISH_DISABLE_AUTO_SSH=1 $PY - <<'EOF'
import json, os
from euclid_polish.ensemble import EnsembleModel, MemberTrainSpec
S = os.environ["S"]
spec = MemberTrainSpec(name="member_900", seed=1, target_steps=20, run_steps=20, loss_norm="l2",
                       asinh_knees=(0.1, 1, 10, 100, 1000, 10000), knee_loss="balanced",
                       output_knee=10.0, learn_output_knee=True, icnr=True, num_res_blocks=32)
ens = EnsembleModel(f"{S}/ensemble", _models=[])
ens.train_members(f"{S}/records/dirty_train.tfrecord", f"{S}/records/clean_train.tfrecord", [spec],
                  batch_size=4, evaluate_every=10, validate_images=4)
print(json.load(open(f"{S}/ensemble/member_900/origin.json"))["learned_output_knee"])
EOF
```

Expected: two validation windows, each printing `per-knee PSNR (dB): …` and `learned output knee (e⁻): VIS … | Y_E … | J_E … | H_E …`; the final line `{'init_e': 10.0, 'min_e': 0.1, 'max_e': 10000.0}`; no `output_knee` key in `origin.json`.

- [ ] **Step 3: Reload and super-resolve a real test field**

```bash
S=$S KMP_DUPLICATE_LIB_OK=TRUE TF_CPP_MIN_LOG_LEVEL=2 EUCLID_POLISH_DISABLE_AUTO_SSH=1 $PY - <<'EOF'
import os
import numpy as np
from euclid_polish.image.tfio import read_images
from euclid_polish.model import Model
from euclid_polish.training.models.output_knee import learned_output_knees
S = os.environ["S"]
m = Model(f"{S}/ensemble/member_900")
print("learned:", m._learn_output_knee, "depth:", m._num_res_blocks,
      "knees:", learned_output_knees(m._tf_model), "kw:", m._knee_kw())
R = "data/_fasrc_cache/n/netscratch/lconnor_lab/Lab/abelotserkovtsev/EuclidPolish/data/images/records_v2"
lr = np.asarray(read_images(R + "/dirty_test.tfrecord", num_images=1)[0].data, np.float32)
sr = m.upsample_array(lr)
print(sr.shape, bool(np.isfinite(sr).all()), float(sr[..., 0].sum()), float(lr[..., 0].sum()))
EOF
```

Expected: `learned: True depth: 32 knees: [~10 ~10 ~10 ~10] kw: {'knees': (...), 'output_electrons': True}`, then `(510, 510, 4) True <sum> <sum>` (an untrained member's sums need not match).

---

### Task 8: Pull FASRC to `main` and submit the member

- [ ] **Step 1: Bring FASRC's checkout to `main`**

```bash
ssh -S /tmp/euclid-polish-fasrc.sock -o BatchMode=yes abelotserkovtsev@login.rc.fas.harvard.edu \
  'cd /n/holylabs/lconnor_lab/Lab/abelotserkovtsev/EuclidPolish && git status --short --untracked-files=no && git pull --ff-only -q && git rev-parse --short HEAD'
git rev-parse --short HEAD
```

Expected: no tracked modifications listed; FASRC's short HEAD equals the local one. (If the socket is missing, reopen it as in the `reference_fasrc_connect` memory.)

- [ ] **Step 2: Confirm the lane is free**

```bash
cat ~/.euclid_polish/fasrc_queue.json
ssh -S /tmp/euclid-polish-fasrc.sock -o BatchMode=yes abelotserkovtsev@login.rc.fas.harvard.edu 'squeue -u abelotserkovtsev -h -o "%i %j %T"'
```

Expected: `"items": []`, `"active_jobid": null`, and no running EuclidPolish job. If anything is pending or running, STOP and report; do not queue (the 2026-09-27 consoles would rebuild a queued command without the new member-spec key).

- [ ] **Step 3: Submit through the `ensemble_train` step**

Member 196's recipe from job 48107719 plus `"learn_output_knee": true`, 100k steps, one member, the same resources.

```bash
cd /Users/alarion239/Desktop/EuclidPolish && KMP_DUPLICATE_LIB_OK=TRUE EUCLID_POLISH_DISABLE_AUTO_SSH=1 $PY - <<'EOF'
import json, os
import pandas as pd
from euclid_polish.web import fasrc_jobs, fasrc_queue, remote
from euclid_polish.web.app import create_app
remote.connect_from_config()
assert not fasrc_queue.QUEUE.active_is_running(fasrc_jobs.DB), "lane busy: do not queue"
log = pd.read_csv(os.path.expanduser("~/.euclid_polish/fasrc_job_log.csv"), low_memory=False)
prev = json.loads(log[log.jobid.astype(str) == "48107719"].iloc[0].params_json)
recipe = json.loads(prev["member_spec"])[1]            # member_196's entry
assert recipe["output_knee"] == 10 and recipe["knee_loss"] == "balanced"
recipe = {k: v for k, v in recipe.items() if k != "seed"}
recipe["learn_output_knee"] = True
skip = {"array_count", "member_names", "base_seed", "step_id", "member_spec", "count", "steps"}
form = {k: v for k, v in prev.items() if not k.startswith("_") and k not in skip}
form.update({"confirm": "yes", "n_cpus": "16", "n_gpus": "1", "memory": "32G",
             "time_limit": "3:00:00", "mode": "add", "count": "1", "steps": "100000",
             "array_max_parallel": "1", "member_spec": json.dumps([recipe])})
app = create_app()
app.config["TESTING"] = True                           # no queue ticker thread
with app.test_client() as client:
    r = client.post("/api/fasrc/steps/ensemble_train/submit",
                    data={k: str(v) for k, v in form.items()})
    print(r.status_code, json.dumps(r.get_json(), indent=1)[:1500])
EOF
```

Expected: `200` with `"ok": true` and a `jobid`. Any `queued: true` is a failure of Step 2's check — cancel it with `POST /api/fasrc/queue/remove` and report.

- [ ] **Step 4: Verify the submitted job**

Replace `<JOBID>` with the `jobid` Step 3 printed.

```bash
ssh -S /tmp/euclid-polish-fasrc.sock -o BatchMode=yes abelotserkovtsev@login.rc.fas.harvard.edu \
  'squeue -j <JOBID> -h -o "%i %j %T %P %l %C %m"; cd /n/holylabs/lconnor_lab/Lab/abelotserkovtsev/EuclidPolish && f=$(ls -t logs/pipeline/ensemble-train-*.sh | head -1) && grep -E "SBATCH --(partition|gres|cpus|mem|time)|train_ensemble.py" "$f" | head -12'
```

Expected: the job PENDING or RUNNING on `gpu`, `--cpus-per-task=16`, `--mem=32G`, `--time=3:00:00`, `--gres=gpu:1`; the command line has `--steps 100000` and a `--member-spec` containing `"learn_output_knee": true`, `"output_knee": 10`, `"num_res_blocks": 32`.

- [ ] **Step 5: Record it**

Update the `project_multi_knee_members` memory with the job id, the member name, and that the run is the learned-knee twin of 196 (depth 32, 100k steps, balanced 6-knee loss).
