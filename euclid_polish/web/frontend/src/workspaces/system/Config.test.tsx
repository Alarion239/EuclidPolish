/* System › Config against a mocked Flask: dirty-only saves with
 * base_version, the 409 conflict rebase (re-homed from the legacy
 * pages/contracts.test.tsx), defaults + reset, "used by" chips, URL filters
 * and validation. */
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import type { ReactElement } from "react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { queryClient } from "../../api/query";
import { UiProvider, resetConfirm } from "../../ui";
import { ConfigKnobsLink } from "../shared/ConfigKnobsLink";
import Config from "./tabs/Config";

type Reply = { status?: number; body: unknown };
let routes: Record<string, (form: Record<string, string>) => Reply>;
let calls: { url: string; method: string; form: Record<string, string> }[];

const CONFIG = {
  vis_pixels: 301, n_train: 1000, n_valid: 100, n_test: 100, hr_image_size: 600,
  psf_warp_prob: 0.2, plateau_lr_enabled: 0, plateau_lr_metric: "combined_loss",
};
const DEFAULTS = { ...CONFIG, vis_pixels: 511, n_train: 6400 };
const SCHEMA = {
  defaults: DEFAULTS,
  types: { vis_pixels: "int", n_train: "int", n_valid: "int", n_test: "int", hr_image_size: "int",
    psf_warp_prob: "float", plateau_lr_enabled: "int", plateau_lr_metric: "str" },
  used_by: { vis_pixels: ["download_euclid_cutouts", "extract_euclid_psf"], n_train: ["synthetic_generate"],
    psf_warp_prob: ["synthetic_generate", "ensemble_train"] },
  steps: { download_euclid_cutouts: "Euclid cutouts", extract_euclid_psf: "Extract ePSF",
    synthetic_generate: "Synthetic generate", ensemble_train: "Ensemble train" },
};

const formOf = (body: BodyInit | null | undefined): Record<string, string> => {
  const out: Record<string, string> = {};
  if (body instanceof FormData) body.forEach((v, k) => { out[k] = String(v); });
  return out;
};

beforeEach(() => {
  calls = [];
  routes = { "GET /api/config": () => ({ body: { ok: true, config: CONFIG, version: "v1", ...SCHEMA } }) };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input);
    const method = init.method ?? "GET";
    const form = formOf(init.body);
    calls.push({ url, method, form });
    const r = routes[`${method} ${url}`]?.(form) ?? { status: 404, body: { ok: false, error: `no route ${url}` } };
    return new Response(JSON.stringify(r.body), { status: r.status ?? 200 });
  }));
  queryClient.clear();
});
afterEach(() => { act(() => resetConfirm()); queryClient.clear(); });

let search = "";
function Where() { search = useLocation().search; return null; }
const show = (el: ReactElement, url = "/system/config") => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <UiProvider>{el}<Where /></UiProvider>
    </MemoryRouter>
  </QueryClientProvider>,
);
const posts = (url: string) => calls.filter((c) => c.method === "POST" && c.url === url);
const saveButton = () => screen.getByRole("button", { name: /^Save/ });

describe("System › Config", () => {
  it("posts only the edited fields with base_version", async () => {
    routes["POST /api/config/save"] = (form) => ({ body: { ok: true, config: { ...CONFIG, n_train: Number(form.n_train) }, version: "v2", note: null } });
    show(<Config />);
    fireEvent.change(await screen.findByLabelText("Train scenes"), { target: { value: "2000" } });
    expect(saveButton().textContent).toContain("1");
    fireEvent.click(saveButton());
    await waitFor(() => expect(posts("/api/config/save")).toHaveLength(1));
    expect(posts("/api/config/save")[0].form).toEqual({ n_train: "2000", base_version: "v1" });
    expect(await screen.findByText("Saved 1 field")).toBeTruthy();
  });

  it("shows a 409 conflict and rebases on the server values, keeping other edits", async () => {
    let n = 0;
    routes["POST /api/config/save"] = (form) => {
      n += 1;
      if (n === 1) {
        return { status: 409, body: {
          ok: false, code: "config_conflict", error: "the config changed since you loaded it: n_train",
          conflicts: { n_train: { base: 1000, current: 5000 } }, config: { ...CONFIG, n_train: 5000 }, version: "v9",
        } };
      }
      return { body: { ok: true, config: { ...CONFIG, n_train: 5000, n_valid: Number(form.n_valid) }, version: "v10" } };
    };
    show(<Config />);
    fireEvent.change(await screen.findByLabelText("Train scenes"), { target: { value: "2000" } });
    fireEvent.change(screen.getByLabelText("Validate scenes"), { target: { value: "150" } });
    fireEvent.click(saveButton());
    expect(await screen.findByText("The config changed since you loaded it")).toBeTruthy();
    expect(screen.getByText("n_train: yours 2000 · now 5000")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Take server values" }));
    expect((screen.getByLabelText("Train scenes") as HTMLInputElement).value).toBe("5000");
    expect((screen.getByLabelText("Validate scenes") as HTMLInputElement).value).toBe("150");
    fireEvent.click(saveButton());
    await waitFor(() => expect(posts("/api/config/save")).toHaveLength(2));
    expect(posts("/api/config/save")[1].form).toEqual({ n_valid: "150", base_version: "v9" });
  });

  it("shows the default and resets a changed field to it (a dirty edit, saved explicitly)", async () => {
    show(<Config />);
    await screen.findByLabelText("Train scenes");
    fireEvent.click(screen.getByRole("button", { name: "Reset Train scenes to 6400" }));
    expect((screen.getByLabelText("Train scenes") as HTMLInputElement).value).toBe("6400");
    expect(posts("/api/config/save")).toHaveLength(0);
    expect(saveButton().textContent).toContain("1");
  });

  it("names the FASRC steps each field is injected into: once per group, extras per field", async () => {
    show(<Config />);
    await screen.findByLabelText("Warp probability");
    // every field of the PSF group feeds both steps: listed once, in the group's head
    const psf = screen.getByRole("region", { name: "PSF distribution & saturation" });
    const head = within(psf).getByLabelText("PSF distribution & saturation is injected into");
    expect(within(head).getByText("Synthetic generate")).toBeTruthy();
    expect(within(head).getByText("Ensemble train")).toBeTruthy();
    const warp = screen.getByLabelText("Warp probability").closest(".cfg-field") as HTMLElement;
    expect(within(warp).queryByText("Synthetic generate")).toBeNull();
    // the scenes group shares no step (n_valid feeds none): n_train lists its own
    const nTrain = screen.getByLabelText("Train scenes").closest(".cfg-field") as HTMLElement;
    expect(within(nTrain).getByText("Synthetic generate")).toBeTruthy();
    expect(within(screen.getByLabelText("Validate scenes").closest(".cfg-field") as HTMLElement)
      .queryByText("Synthetic generate")).toBeNull();
  });

  it("shows a field's default only when the value differs from it", async () => {
    show(<Config />);
    const nValid = (await screen.findByLabelText("Validate scenes")).closest(".cfg-field") as HTMLElement;
    expect(within(nValid).queryByText(/default/)).toBeNull();             // 100 = default
    const nTrain = screen.getByLabelText("Train scenes").closest(".cfg-field") as HTMLElement;
    expect(within(nTrain).getByText("6400")).toBeTruthy();                // 1000 ≠ 6400
  });

  it("shortens a long default for display but resets to its exact value", async () => {
    routes["GET /api/config"] = () => ({ body: { ok: true, config: { ...CONFIG, psf_warp_prob: 0.25 }, version: "v1",
      ...SCHEMA, defaults: { ...DEFAULTS, psf_warp_prob: 1 / 3 } } });
    show(<Config />);
    const warp = (await screen.findByLabelText("Warp probability")).closest(".cfg-field") as HTMLElement;
    expect(within(warp).getByText("0.333333").getAttribute("title")).toBe(String(1 / 3));
    fireEvent.click(within(warp).getByRole("button", { name: "Reset Warp probability to 0.333333" }));
    expect((screen.getByLabelText("Warp probability") as HTMLInputElement).value).toBe(String(1 / 3));
  });

  it("keeps the filters in the URL and narrows the fields", async () => {
    show(<Config />);
    await screen.findByLabelText("Train scenes");
    fireEvent.change(screen.getByLabelText("Filter fields"), { target: { value: "warp" } });
    await waitFor(() => expect(search).toBe("?q=warp"));
    expect(screen.queryByLabelText("Train scenes")).toBeNull();
    expect(screen.getByLabelText("Warp probability")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Filter fields"), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: /Changed from default/ }));
    await waitFor(() => expect(search).toBe("?changed=1"));
    expect(screen.getByLabelText("Train scenes")).toBeTruthy();       // 1000 ≠ 6400
    expect(screen.queryByLabelText("Validate scenes")).toBeNull();    // at its default
  });

  it("shows only the empty state (no reset footer) when no field matches", async () => {
    show(<Config />, "/system/config?q=zzz-nothing");
    expect(await screen.findByText("No field matches")).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Set every field to its default/ })).toBeNull();
  });

  it("reads an unknown ?group= as all groups", async () => {
    show(<Config />, "/system/config?group=training-lr");
    expect(await screen.findByLabelText("Train scenes")).toBeTruthy();
    expect(screen.queryByText("No field matches")).toBeNull();
  });

  it("refuses to save an out-of-range value and says why", async () => {
    show(<Config />);
    fireEvent.change(await screen.findByLabelText("Warp probability"), { target: { value: "1.5" } });
    expect(await screen.findByText("must be ≤ 1")).toBeTruthy();
    expect((saveButton() as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText("1 invalid")).toBeTruthy();
  });

  it("shows the server's error when the config cannot be read", async () => {
    routes["GET /api/config"] = () => ({ status: 400, body: { ok: false, error: "job_config.json is not valid JSON" } });
    show(<Config />);
    expect(await screen.findByText("job_config.json is not valid JSON")).toBeTruthy();
  });
});

describe("System › Config (regrouping)", () => {
  const FULL = { ...CONFIG, galaxy_density_arcmin2: 151.5032458303819, star_density_arcmin2: 5.2, asinh_scale: 1000 };
  beforeEach(() => {
    routes["GET /api/config"] = () => ({ body: { ok: true, config: FULL, version: "v1", ...SCHEMA,
      defaults: { ...DEFAULTS, galaxy_density_arcmin2: 50, star_density_arcmin2: 5.2, asinh_scale: 1000 },
      types: { ...SCHEMA.types, galaxy_density_arcmin2: "float", star_density_arcmin2: "float", asinh_scale: "float" } } });
    routes["GET /api/realism/overview"] = () => ({ body: { items: [
      { id: "star-prior", facts: { is_active: true, density_arcmin2: 5.0842 } },
    ] } });
  });

  it("links each group to the tab where its effect is judged", async () => {
    show(<Config />);
    const scenes = await screen.findByRole("region", { name: "Synthetic scenes" });
    expect(within(scenes).getByRole("link", { name: "judged on Synthetic › Records" }).getAttribute("href")).toBe("/synthetic/records");
    const plateau = screen.getByRole("region", { name: "Training · plateau guard" });
    expect(within(plateau).getByRole("link", { name: "judged on Models › Train" }).getAttribute("href")).toBe("/models/train");
    const psf = screen.getByRole("region", { name: "PSF distribution & saturation" });
    expect(within(psf).getByRole("link", { name: "judged on Synthetic › PSF" }).getAttribute("href")).toBe("/synthetic/psf");
  });

  it("shows star density read-only, set by the active stellar prior, and rounds the galaxy density", async () => {
    show(<Config />);
    expect(await screen.findByText("set by the active stellar prior (5.08 arcmin⁻²)")).toBeTruthy();
    expect(screen.queryByLabelText("Star density")).toBeNull();
    expect((screen.getByLabelText("Galaxy density") as HTMLInputElement).value).toBe("151.5");
    expect(screen.getByRole("button", { name: /^Save/ }).hasAttribute("disabled")).toBe(true);   // rounding is not an edit
    expect(screen.queryByText("Display")).toBeNull();            // the dead Display group is gone
    expect(screen.queryByLabelText(/asinh scale/i)).toBeNull();
  });

  it("asks before setting every field to its default", async () => {
    show(<Config />);
    await screen.findByLabelText("Train scenes");
    fireEvent.click(screen.getByRole("button", { name: "Set every field to its default…" }));
    const dlg = await screen.findByRole("alertdialog", { name: "Set every field to its default?" });
    fireEvent.click(within(dlg).getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull());
    expect((screen.getByLabelText("Train scenes") as HTMLInputElement).value).toBe("1000");
    fireEvent.click(screen.getByRole("button", { name: "Set every field to its default…" }));
    fireEvent.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Set to defaults" }));
    await waitFor(() => expect((screen.getByLabelText("Train scenes") as HTMLInputElement).value).toBe("6400"));
    expect(posts("/api/config/save")).toHaveLength(0);                   // nothing saved yet
  });
});

describe("ConfigKnobsLink (the back-link on the tabs that judge a group)", () => {
  it("says how many of the group's knobs differ from default and links to them in Config", async () => {
    show(<ConfigKnobsLink groups={["scenes"]} />, "/synthetic/records");
    const link = await screen.findByRole("link", { name: "1 knob changed · Edit" });   // n_train 1000 vs 6400
    expect(link.getAttribute("href")).toBe("/system/config?group=scenes&changed=1");
    expect(calls.every((c) => c.method === "GET")).toBe(true);
  });

  it("shows nothing when every knob of the group is at its default", async () => {
    show(<ConfigKnobsLink groups={["plateau"]} />, "/models/train");
    await waitFor(() => expect(calls.some((c) => c.url === "/api/config")).toBe(true));
    expect(screen.queryByRole("link")).toBeNull();
  });
});
