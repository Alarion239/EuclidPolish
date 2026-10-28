/* Models › Universal › Milestones: "sharper/blurrier" wording (never a
 * signed %), and the section drawn from a mocked payload: the summary line,
 * the four charts and a table row per milestone; nothing when no run has
 * milestones. */
import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { queryClient } from "../../api/query";
import { MilestonesSection, milestonesUrl, sharpnessText, type MilestonesPayload } from "./MilestonesSection";

const scan = (a: number) => [0.1, 1, 10].map((multiple, i) => ({ multiple, sb: 28 - i, arcsec: a / (i + 1) }));
const payload = (): MilestonesPayload => ({
  members: [{
    member: "member_257",
    steps: [200_000, 8_000_000].map((step, i) => ({
      step, spec: `milestone:member_257@${step}`, mean: 59.2 + 3.3 * i, integrated: [53 + i, 63 + i, 60 + i, 59 + i],
      hours: 0.9 + 36 * i, learned_knees: [0.5, 0.1, 0.1, 0.1], scored_at: "2026-10-27T10:00:00Z",
    })),
    resolution: { H_E: { jwst_filter: "F200W", noise_sb: 25.7, models: {
      "milestone:member_257@200000": scan(1.1), "milestone:member_257@8000000": scan(1.0), production: scan(1.2),
    } } },
  }],
});

let body: unknown;
beforeEach(() => {
  body = payload();
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => new Response(
    JSON.stringify(String(input) === milestonesUrl ? body : {}), { status: 200 })));
  queryClient.clear();
});
afterEach(() => queryClient.clear());

const show = () => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={["/models/universal"]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <MilestonesSection references={[{ label: "production (#238)", mean: 62.13, integrated: [56.5, 66.3, 63.2, 62.4] }]} />
    </MemoryRouter>
  </QueryClientProvider>,
);

describe("milestones", () => {
  it("says sharper or blurrier, never a signed percentage", () => {
    expect(sharpnessText(1.0, 1.2)).toBe("16.7% sharper");
    expect(sharpnessText(1.2, 1.0)).toBe("20.0% blurrier");
    expect(sharpnessText(1.0, 1.0)).toBe("same");
    expect(sharpnessText(null, 1.0)).toBe("—");
  });

  it("draws the newest run's milestones: summary, four charts and a row per milestone", async () => {
    show();
    expect(await screen.findByRole("heading", { name: "Milestones of a long run" })).toBeTruthy();
    expect(document.querySelector(".ui-summary")?.textContent).toMatch(
      /#257 at 8M steps \(36\.9 h\): ∫PSNR 62\.50 dB \(\+0\.37 dB vs production\); H at 1σ: 16\.7% sharper than production on JWST\./);
    for (const name of ["Milestone integrated PSNR vs training steps", "Resolution vs surface brightness, H",
      "Resolution at 1 sigma vs training steps, H", "Learned output knee vs training steps"]) {
      expect(screen.getByRole("figure", { name })).toBeTruthy();
    }
    const table = screen.getByRole("table", { name: "Milestones" });
    expect(table.querySelectorAll("tbody tr")).toHaveLength(2);
    expect(table.textContent).toContain("0.500″ · 16.7% sharper");
  });

  it("draws nothing when no run has milestones", async () => {
    body = { members: [] };
    const { container } = show();
    await waitFor(() => expect(vi.mocked(fetch)).toHaveBeenCalled());
    expect(container.querySelector("section")).toBeNull();
  });
});
