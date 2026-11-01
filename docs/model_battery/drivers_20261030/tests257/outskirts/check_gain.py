"""Reproducibility of the real-sky gradient pipeline: 238 re-run on the first 2 gradient tiles vs the 10-28 arrays."""
import numpy as np

U = "/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/d1a24d4c-2869-4369-a6ae-08d92714c971/scratchpad/underflux/out"
W = "/private/tmp/claude-502/-Users-alarion239-Desktop-EuclidPolish/6d444c86-12c6-4a7f-b4eb-7b73fa2e1c66/scratchpad/tests257/outskirts/runs"
pick = np.load(f"{U}/grad_tiles.npy")[:2]
for name, new, old in (("G", np.load(f"{W}/gain_238_check_G.npy"), np.load(f"{U}/G_member_238.npy", mmap_mode="r")[:2]),
                       ("A", np.load(f"{W}/gain_238_check_A.npy"), np.load(f"{U}/A_member_238.npy", mmap_mode="r")[pick]),
                       ("R", np.load(f"{W}/gain_238_check_R.npy"), np.load(f"{U}/R_member_238.npy", mmap_mode="r")[pick])):
    old = np.asarray(old)
    print(name, "identical", np.array_equal(new, old), "max|Δ|", float(np.max(np.abs(new - old))),
          "max|old|", float(np.max(np.abs(old))))
