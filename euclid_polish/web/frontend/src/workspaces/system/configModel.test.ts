import { describe, expect, it } from "vitest";
import { FIELDS } from "./configFields";
import {
  changedKnobs, displayValue, commonSteps, dirtyFields, fieldError, filterFields, isDefault, orderedFields, rebase, saveBody, toForm, type Conflict,
} from "./configModel";

const CONFIG = { vis_pixels: 511, n_train: 6400, n_valid: 100, psf_warp_prob: 0.2, plateau_lr_enabled: 0, plateau_lr_metric: "combined_loss" };
const DEFAULTS = { ...CONFIG, psf_warp_prob: 0.5 };

describe("form state", () => {
  it("keeps every value as the string the input shows", () => {
    expect(toForm(CONFIG)).toEqual({
      vis_pixels: "511", n_train: "6400", n_valid: "100", psf_warp_prob: "0.2", plateau_lr_enabled: "0",
      plateau_lr_metric: "combined_loss",
    });
    expect(toForm({ x: null })).toEqual({ x: "" });
  });

  it("lists only the edited fields and posts them with base_version", () => {
    const loaded = toForm(CONFIG);
    const form = { ...loaded, n_train: "2000", psf_warp_prob: "0.2" };
    expect(dirtyFields(form, loaded)).toEqual(["n_train"]);
    expect(saveBody(form, loaded, "v1")).toEqual({ n_train: "2000", base_version: "v1" });
    expect(saveBody(form, loaded, null)).toEqual({ n_train: "2000" });
  });

  it("rebases on a 409: the server's values for conflicting fields, my other edits kept", () => {
    const loaded = toForm(CONFIG);
    const form = { ...loaded, n_train: "2000", n_valid: "150" };
    const conflict: Conflict = { fields: { n_train: { base: 6400, current: 5000 } }, config: { ...CONFIG, n_train: 5000 }, version: "v9" };
    const next = rebase(form, loaded, conflict);
    expect(next.loaded.n_train).toBe("5000");
    expect(next.form.n_train).toBe("5000");
    expect(next.form.n_valid).toBe("150");
    expect(next.version).toBe("v9");
    expect(saveBody(next.form, next.loaded, next.version)).toEqual({ n_valid: "150", base_version: "v9" });
  });
});

describe("defaults and validation", () => {
  it("compares numbers by value, strings verbatim", () => {
    expect(isDefault("psf_warp_prob", "0.50", DEFAULTS)).toBe(true);
    expect(isDefault("psf_warp_prob", "0.2", DEFAULTS)).toBe(false);
    expect(isDefault("plateau_lr_metric", "combined_loss", DEFAULTS)).toBe(true);
    expect(isDefault("unknown", "1", DEFAULTS)).toBe(true);
  });

  it("flags non-numbers, out-of-range values and fractions in int fields", () => {
    expect(fieldError("n_train", "abc")).toMatch(/number/);
    expect(fieldError("n_train", "0")).toMatch(/≥ 1/);
    expect(fieldError("psf_warp_prob", "1.5")).toMatch(/≤ 1/);
    expect(fieldError("n_train", "10.5")).toMatch(/whole/);
    expect(fieldError("n_train", "")).toMatch(/required/);
    expect(fieldError("n_train", "2000")).toBeNull();
    expect(fieldError("plateau_lr_metric", "combined_loss")).toBeNull();
  });
});

describe("filtering and ordering", () => {
  const form = toForm({ ...DEFAULTS, psf_warp_prob: 0.2 });
  const loaded = toForm(DEFAULTS);

  it("orders the server's fields by the known groups, unknown ones last", () => {
    const names = orderedFields(["zzz_new", "n_train", "vis_pixels"]);
    expect(names.map((f) => f.name)).toEqual(["vis_pixels", "n_train", "zzz_new"]);
    expect(names[2].group).toBe("other");
  });

  it("filters by text (label, name, hint), group, changed-from-default and edited", () => {
    const all = orderedFields(Object.keys(DEFAULTS));
    const pick = (opts: Parameters<typeof filterFields>[1]) => filterFields(all, opts, form, loaded, DEFAULTS).map((f) => f.name);
    expect(pick({ q: "warp" })).toEqual(["psf_warp_prob"]);
    expect(pick({ q: "odd" })).toEqual(["vis_pixels"]);                  // matched in the hint
    expect(pick({ group: "scenes" })).toEqual(["n_train", "n_valid"]);
    expect(pick({ changed: true })).toEqual(["psf_warp_prob"]);
    expect(pick({ edited: true })).toEqual(["psf_warp_prob"]);
    expect(pick({})).toHaveLength(all.length);
  });

  it("finds the FASRC steps every field of a group feeds (shown once per group)", () => {
    const fields = orderedFields(["psf_warp_prob", "psf_warp_sigma", "saturation_mask_prob"]);
    const usedBy = {
      psf_warp_prob: ["synthetic_generate", "ensemble_train"],
      psf_warp_sigma: ["synthetic_generate", "ensemble_train"],
      saturation_mask_prob: ["ensemble_train", "synthetic_generate", "other_step"],
    };
    expect(commonSteps(fields, usedBy)).toEqual(["synthetic_generate", "ensemble_train"]);
    expect(commonSteps(fields, { ...usedBy, psf_warp_sigma: [] })).toEqual([]);   // a local-only field
    expect(commonSteps([], usedBy)).toEqual([]);
  });

  it("describes every JobConfig field it lists exactly once", () => {
    const names = FIELDS.map((f) => f.name);
    expect(new Set(names).size).toBe(names.length);
  });
});

describe("display (console regrouping)", () => {
  it("rounds a fitted value for display and never marks it edited", () => {
    expect(displayValue("galaxy_density_arcmin2", 151.5032458303819)).toBe("151.5");
    expect(displayValue("n_train", 6400)).toBe("6400");
    const loaded = toForm({ galaxy_density_arcmin2: 151.5032458303819, n_train: 6400 });
    expect(loaded.galaxy_density_arcmin2).toBe("151.5");
    expect(dirtyFields({ ...loaded }, loaded)).toEqual([]);
    expect(isDefault("galaxy_density_arcmin2", "151.5", { galaxy_density_arcmin2: 151.5032458303819 })).toBe(true);
  });
  it("leaves the dead fields out and has no Display group", () => {
    expect(orderedFields(["asinh_scale", "n_train"]).map((f) => f.name)).toEqual(["n_train"]);
    expect(FIELDS.some((f) => f.name === "asinh_scale")).toBe(false);
    expect(FIELDS.find((f) => f.name === "star_density_arcmin2")?.readOnly).toBe(true);
  });
});

describe("changed knobs per group (the 'N knobs changed · Edit' back-link)", () => {
  it("counts the group's editable knobs off their default, as the editor rounds them", () => {
    const config = { n_train: 1000, n_valid: 100, hr_image_size: 600, galaxy_density_arcmin2: 151.5000001,
      star_density_arcmin2: 9, psf_warp_prob: 0.2 };
    const defaults = { n_train: 6400, n_valid: 100, hr_image_size: 600, galaxy_density_arcmin2: 151.5032458,
      star_density_arcmin2: 5.08, psf_warp_prob: 0.5 };
    expect(changedKnobs(config, defaults, ["scenes"])).toBe(1);            // n_train only (galaxy density rounds equal)
    expect(changedKnobs(config, defaults, ["stars"])).toBe(0);             // read-only star density is not a knob
    expect(changedKnobs(config, defaults, ["scenes", "psf"])).toBe(2);
    expect(changedKnobs(config, {}, ["scenes"])).toBe(0);                  // no defaults, nothing changed
  });
});
