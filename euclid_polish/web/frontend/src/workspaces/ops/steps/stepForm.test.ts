import { describe, expect, it } from "vitest";
import {
  changedParams, dangerReasons, defaultResources, initialValues, isChanged, paramFacts, resourcesFromRow,
  submitBody, toFormValue, validateParam, validateResources, validateValues, visibleParams,
  type Step, type TaskParam,
} from "./stepForm";

const P = (p: Partial<TaskParam> & { name: string; type: TaskParam["type"] }): TaskParam =>
  ({ default: null, help: "", ...p });

const QUERY: Step = {
  step_id: "euclid_query", label: "Query", needs_gpu: false,
  defaults: { partition: "shared", n_cpus: 1, n_gpus: 0, memory: "4G", time_limit: "30:00" },
  task_params: [
    P({ name: "num_stars", type: "int", default: 10000, min: 1 }),
    P({ name: "magnitude_min", type: "float", default: 18.0 }),
    P({ name: "snr_min", type: "float", default: 50.0, min: 0 }),
  ],
  last_params: null,
};

describe("toFormValue", () => {
  it("serialises typed values like the server's form_value", () => {
    expect(toFormValue(P({ name: "b", type: "bool" }), true)).toBe("1");
    expect(toFormValue(P({ name: "b", type: "bool" }), "yes")).toBe("1");
    expect(toFormValue(P({ name: "b", type: "bool" }), false)).toBe("0");
    expect(toFormValue(P({ name: "f", type: "float" }), 18.0)).toBe("18");
    expect(toFormValue(P({ name: "f", type: "float" }), 0.3)).toBe("0.3");
    expect(toFormValue(P({ name: "j", type: "json" }), [{ loss: "l2" }])).toBe('[{"loss":"l2"}]');
    expect(toFormValue(P({ name: "s", type: "str" }), null)).toBe("");
  });
});

describe("initialValues", () => {
  it("starts from the schema defaults", () => {
    expect(initialValues(QUERY)).toEqual({
      values: { num_stars: "10000", magnitude_min: "18", snr_min: "50" }, source: "defaults" });
  });
  it("prefills from the last successful run, keeping defaults for nulls", () => {
    const step = { ...QUERY, last_params: { num_stars: 200, magnitude_min: null, snr_min: 50 } };
    expect(initialValues(step)).toEqual({
      values: { num_stars: "200", magnitude_min: "18", snr_min: "50" }, source: "last" });
  });
  it("reports defaults when the last run equals them", () => {
    const step = { ...QUERY, last_params: { num_stars: 10000 } };
    expect(initialValues(step).source).toBe("defaults");
  });
  it("never carries one-shot destructive flags over from the last run", () => {
    const step: Step = { ...QUERY, task_params: [...QUERY.task_params!,
      P({ name: "regenerate_catalog", type: "bool", default: false }),
      P({ name: "force", type: "bool", default: false })],
      last_params: { num_stars: 10000, regenerate_catalog: true, force: true } };
    const out = initialValues(step);
    expect(out.values.regenerate_catalog).toBe("0");
    expect(out.values.force).toBe("0");
    expect(out.source).toBe("defaults");
    // An explicit clone still copies them (the danger confirm then asks).
    expect(initialValues(step, { clone: { regenerate_catalog: "1" } }).values.regenerate_catalog).toBe("1");
  });
  it("a clone source wins and ignores unknown keys", () => {
    const out = initialValues({ ...QUERY, last_params: { num_stars: 200 } },
      { clone: { num_stars: "42", other: "x", _star_prior_json: "{}" } });
    expect(out.values.num_stars).toBe("42");
    expect(out.values).not.toHaveProperty("other");
    expect(out.source).toBe("clone");
  });
});

describe("validateParam mirrors TaskParam.parse", () => {
  it("checks ints, floats, ranges and blanks", () => {
    const n = QUERY.task_params![0];
    expect(validateParam(n, "10")).toBeNull();
    expect(validateParam(n, "1.5")).toBe("must be an integer");
    expect(validateParam(n, "0")).toBe("must be ≥ 1");
    expect(validateParam(n, "abc")).toBe("must be an integer");
    expect(validateParam(n, "")).toBeNull();
    expect(validateParam({ ...n, required: true }, " ")).toBe("required");
    expect(validateParam(P({ name: "t", type: "float", min: 0, max: 1 }), "1.5")).toBe("must be ≤ 1");
  });
  it("checks choices, json and bools", () => {
    const c = P({ name: "band", type: "choice", choices: ["VIS", "Y"] });
    expect(validateParam(c, "VIS")).toBeNull();
    expect(validateParam(c, "Z")).toBe("must be one of VIS, Y");
    expect(validateParam(P({ name: "j", type: "json" }), "[1,")).toBe("is not valid JSON");
    expect(validateParam(P({ name: "b", type: "bool" }), "maybe")).toBe("must be on or off");
  });
  it("validateValues skips host-controlled params", () => {
    const errs = validateValues(QUERY.task_params!, { num_stars: "0", magnitude_min: "x", snr_min: "1" }, new Set(["magnitude_min"]));
    expect(errs).toEqual({ num_stars: "must be ≥ 1" });
  });
});

describe("changes and visibility", () => {
  it("compares numerically with the default", () => {
    const m = QUERY.task_params![1];
    expect(isChanged(m, "18.0")).toBe(false);
    expect(isChanged(m, "")).toBe(true);
    expect(isChanged(m, "19")).toBe(true);
    expect(changedParams(QUERY.task_params!, { num_stars: "10000", magnitude_min: "17", snr_min: "50" })
      .map((p) => p.name)).toEqual(["magnitude_min"]);
  });
  it("shows the first N plus every changed or invalid param", () => {
    const params = Array.from({ length: 12 }, (_, i) => P({ name: `p${i}`, type: "int", default: 1 }));
    const values = Object.fromEntries(params.map((p) => [p.name, "1"]));
    values.p10 = "5";
    const { shown, more } = visibleParams(params, values, { limit: 4, errors: { p11: "bad" } });
    expect(shown.map((p) => p.name)).toEqual(["p0", "p1", "p2", "p3", "p10", "p11"]);
    expect(more).toBe(6);
    expect(visibleParams(params, values, { expanded: true }).more).toBe(0);
    expect(visibleParams(params, values, { hidden: new Set(["p0"]), expanded: true }).shown).toHaveLength(11);
  });
});

describe("submitBody", () => {
  it("posts resources, form params, extra params and the confirm token", () => {
    const body = submitBody(QUERY, { num_stars: " 500 ", magnitude_min: "", snr_min: "50" },
      defaultResources(QUERY), { snr_min: 10 }, new Set(["snr_min"]));
    expect(body).toEqual({
      n_cpus: "1", n_gpus: "0", memory: "4G", time_limit: "30:00",
      num_stars: "500", magnitude_min: "", snr_min: "10", confirm: "yes",
    });
  });
  it("forces the fixed CPU/GPU counts", () => {
    const step: Step = { ...QUERY, needs_gpu: true, fixed_cpus: 4, fixed_gpus: 1 };
    const body = submitBody(step, {}, { n_cpus: "64", n_gpus: "8", memory: "1G", time_limit: "1:00:00" });
    expect(body.n_cpus).toBe("4");
    expect(body.n_gpus).toBe("1");
  });
});

describe("resources", () => {
  it("defaults a CPU step to zero GPUs and validates formats", () => {
    expect(defaultResources(QUERY)).toEqual({ n_cpus: "1", n_gpus: "0", memory: "4G", time_limit: "30:00" });
    expect(validateResources(QUERY, { n_cpus: "0", n_gpus: "x", memory: "lots", time_limit: "soon" })).toEqual({
      n_cpus: "a whole number ≥ 1", n_gpus: "a whole number ≥ 0", memory: "e.g. 16G or 512M",
      time_limit: "e.g. 2:00:00 or 1-00:00:00" });
    expect(validateResources(QUERY, { n_cpus: "8", n_gpus: "0", memory: "64G", time_limit: "1-00:00:00" })).toEqual({});
    expect(validateResources({ ...QUERY, fixed_cpus: 1 }, { n_cpus: "", n_gpus: "0", memory: "4G", time_limit: "30:00" })).toEqual({});
  });
  it("clones a past run's requested resources", () => {
    expect(resourcesFromRow(QUERY, { req_cpus: "4", req_gpus: "", req_memory: "8G", req_time_limit: "2:00:00" }))
      .toEqual({ n_cpus: "4", n_gpus: "0", memory: "8G", time_limit: "2:00:00" });
  });
});

describe("danger and facts", () => {
  it("flags destructive booleans", () => {
    const step: Step = { ...QUERY, task_params: [P({ name: "force", type: "bool", default: false, help: "Regenerate every split" })] };
    expect(dangerReasons(step, { force: "1" })).toEqual(["Regenerate every split"]);
    expect(dangerReasons(step, { force: "0" })).toEqual([]);
    expect(dangerReasons(step, { force: "1" }, new Set(["force"]))).toEqual([]);
  });
  it("summarises default and range", () => {
    expect(paramFacts(QUERY.task_params![0])).toBe("default: 10000 · ≥ 1");
    expect(paramFacts(P({ name: "t", type: "float", default: 0.3, min: 0, max: 1 }))).toBe("default: 0.3 · 0 – 1");
    expect(paramFacts(P({ name: "s", type: "int", required: true }))).toBe("default: unset · required");
    expect(paramFacts(P({ name: "b", type: "bool", default: true }))).toBe("default: on");
  });
});
