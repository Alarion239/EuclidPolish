/* The Models loop's pure logic added by the console regrouping (plan
 * "Console phases 2–6", Team M): the Leaderboard status line and its real
 * benchmark, the members waiting on FASRC, the gate share bar, the combiner
 * variant scope, the synthetic stamps and SR→HR recovery, Δm vs LR, the
 * running training batch and the Train submit label. */
import { describe, expect, it } from "vitest";
import { deltaMagText, runningBatches, submitLabel, forwardModelFacts, lrScheduleText, universalJobMembers } from "./model";

describe("members waiting on FASRC", () => {

  it("reads a member's universal recipe from the run-wide params too", () => {
    const j = (params: Record<string, unknown>) => ({ member_names: ["member_1", "member_2"], params });
    expect([...universalJobMembers(j({ asinh_knees: "0.1,1,10", output_knee: "10" }))]).toEqual(["member_1", "member_2"]);
    expect([...universalJobMembers(j({ asinh_knees: "0.1,1,10", learn_output_knee: "1" }))]).toEqual(["member_1", "member_2"]);
    expect([...universalJobMembers(j({ asinh_knees: "0.1,1,10", output_knee: "", learn_output_knee: "0" }))]).toEqual([]);
    expect([...universalJobMembers(j({ asinh_knees: "", output_knee: "10" }))]).toEqual([]);
    // a member_spec entry overrides the run-wide recipe
    expect([...universalJobMembers(j({ asinh_knees: "0.1,1,10", output_knee: "10", member_spec: JSON.stringify([{ asinh_knees: null }]) }))])
      .toEqual(["member_2"]);
    expect([...universalJobMembers(j({ member_spec: "not json" }))]).toEqual([]);
  });
});

describe("synthetic stamps and SR → HR recovery", () => {

  it("states SR's magnitude change vs LR, warn-toned beyond 0.1 mag", () => {
    expect(deltaMagText(100, 90)).toEqual({ text: "Δm +0.11 (flux ×0.90)", warn: true });
    expect(deltaMagText(100, 99)).toEqual({ text: "Δm +0.01 (flux ×0.99)", warn: false });
    expect(deltaMagText(100, 120)?.text).toBe("Δm −0.20 (flux ×1.20)");
    expect(deltaMagText(0, 5)).toBeNull();
    expect(deltaMagText(null, 5)).toBeNull();
  });
});

describe("train", () => {
  it("names the count in the submit button", () => {
    expect(submitLabel("add", 4)).toBe("Submit 4 members to SLURM");
    expect(submitLabel("add", 1)).toBe("Submit 1 member to SLURM");
    expect(submitLabel("fork", 2)).toBe("Submit 2 forks to SLURM");
    expect(submitLabel("continue", 3)).toBe("Continue 3 members on SLURM");
    expect(submitLabel("continue", 0)).toBe("Continue members on SLURM");
  });

  it("reads the running training batches from the live SLURM rows, with their members", () => {
    const b = runningBatches(
      [{ jobid: "48107719", state: "RUNNING", step_id: "ensemble_train", progress_step: 10650, progress_total: 70000, time: "1:02:00", time_limit: "3:00:00" },
        { jobid: "5", state: "RUNNING", step_id: "synthetic_generate" }],
      [{ jobid: "48107719", mode: "add", member_names: ["member_199", "member_200", "member_201", "member_202"] }],
    );
    expect(b).toEqual([{ jobid: "48107719", state: "RUNNING", members: ["member_199", "member_200", "member_201", "member_202"],
      text: "Job 48107719 · members 199–202 · running · step 10,650 / 70,000 (15%) · 1:02:00 of 3:00:00" }]);
    expect(runningBatches([{ jobid: "7", state: "PENDING", step_id: "ensemble_train", reason: "Priority" }], [])[0].text)
      .toBe("Job 7 · pending (Priority)");
  });

  it("lists the forward-model values the batch takes from System › Config", () => {
    const facts = forwardModelFacts({ psf_warp_prob: 1, psf_warp_alpha_max: 5, psf_warp_sigma: 3, saturation_mask_prob: 0.5, lr_peak: 2e-4 });
    expect(facts.map((f) => [f.label, f.value, f.unit ?? null])).toEqual([
      ["PSF warp probability", "1", null], ["PSF warp α max", "5", null], ["PSF warp σ", "3", "HR px"], ["Saturation mask probability", "0.5", null],
    ]);
    expect(lrScheduleText({ lr_peak: 0.0002, lr_final: 1e-6, lr_warmup_steps: 2000, plateau_lr_enabled: false }))
      .toBe("LR warmup 2,000 steps to 2e-4, then cosine to 1e-6; plateau guard off");
    expect(lrScheduleText({})).toBeNull();
  });
});
