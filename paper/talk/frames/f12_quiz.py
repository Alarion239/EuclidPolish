"""F12 / F15: the quiz "which four are real?" and its payoff.

Eight 25.5" fields, as in the paper's euclid_vs_synthetic figure: four real Q1 MER archive cutouts and
four synthetic LR fields from the v8 test records (11-22 regeneration with the new NISP holes, f280290e), shuffled and lettered A-H. Payoff: every field
POLISHed by the production model, and the clean truth next to the synthetic ones. One stretch for all
Euclid tiles (the collage's: white = 99.7th pct of the real panels, knee white/40), reused for POLISHed
and truth on the 0.05" grid (flux per pixel / 4)."""
import json
import os
import sys

sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish/paper_figures/generation_collages")
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
os.chdir("/Users/alarion239/Desktop/EuclidPolish")
import numpy as np
import pathlib

import build
from build import calibrated, intensity, lupton, real_field, syn

build.RECORDS = pathlib.Path("/Users/alarion239/Desktop/EuclidPolish/data/records_v8_test")

from euclid_polish.eval.ensemble_infer import load_eval_ensemble, sr_from_model
from style import OUT, save_panel

REAL = ["023", "141", "066", "192"]
SYN = [43, 32, 33, 83]                                           # v8 test fields (v6 quiz used 43, 35, 19, 4)
ORDER = ["r0", "s0", "s1", "r1", "s2", "r2", "r3", "s3"]          # fixed shuffle -> letters A..H
LETTERS = "ABCDEFGH"


def crop255(c):
    n = min(c.shape[0], c.shape[1])
    o0, o1 = (c.shape[0] - 255) // 2, (c.shape[1] - 255) // 2
    return c[o0:o0 + 255, o1:o1 + 255]


lr = {f"r{i}": crop255(np.asarray(real_field(x), np.float64)) for i, x in enumerate(REAL)}
lr.update({f"s{i}": crop255(syn("dirty", x)) for i, x in enumerate(SYN)})
truth = {f"s{i}": np.asarray(syn("clean", x), np.float64)[:510, :510] for i, x in enumerate(SYN)}
model = load_eval_ensemble(log=print)
sr = {k: np.asarray(sr_from_model(model, v.astype(np.float32))[1], np.float64) for k, v in lr.items()}

cal_lr = {k: calibrated(v) for k, v in lr.items()}
white = np.percentile(np.concatenate([intensity(cal_lr[f"r{i}"]).ravel() for i in range(4)]), 99.7)
up = lambda c: np.repeat(np.repeat(c, 2, 0), 2, 1) / 4.0
legend = ("VIS", "Y+J", "H")
key = {}
for letter, k in zip(LETTERS, ORDER):
    real = k.startswith("r")
    key[letter] = ("real Q1 " + REAL[int(k[1])]) if real else ("synthetic v8 test " + str(SYN[int(k[1])]))
    save_panel(lupton(up(cal_lr[k]), white=white / 4, q=40), f"F12_quiz_{letter}_1_euclid.png",
               label=letter, legend=legend, interp="nearest", envelope=True)
    save_panel(lupton(calibrated(sr[k]), white=white / 4, q=40), f"F15_quiz_{letter}_2_polished.png",
               label=f"{letter} · POLISHed", legend=legend, bar_arcsec=5.0)
    if not real:
        save_panel(lupton(calibrated(truth[k]), white=white / 4, q=40), f"F15_quiz_{letter}_3_truth.png",
                   label=f"{letter} · TNG truth", legend=legend, bar_arcsec=5.0)
json.dump(key, open(os.path.join(OUT, "F12_quiz_answer_key.json"), "w"), indent=1)
print(key)
