import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Job, JobStatus } from "./api/jobs";
import { JobProgressView } from "./jobs";
import uiCss from "./ui/ui.css?raw";

const job = (status: JobStatus, extra: Partial<Job> = {}): Job => ({
  job_id: "j1", label: "demo job", status, started: 0, finished: null, duration: 3,
  error: null, log: null, log_truncated: false, progress: null, kind: null,
  cancellable: true, result: null, ...extra,
} as Job);

describe("JobProgressView", () => {
  it("tones the panel of every finished status (a stylesheet rule exists for each modifier)", () => {
    for (const status of ["done", "failed", "cancelled"] as const) {
      const { container, unmount } = render(<JobProgressView job={job(status)} />);
      expect(container.querySelector(`.job-panel--${status}`), status).toBeTruthy();
      expect(uiCss, `.job-panel--${status} rule`).toMatch(
        new RegExp(`\\.job-panel--${status}\\b[^{]*\\{[^}]*border-color`),
      );
      unmount();
    }
  });

  it("shows a cancelled job with a warn badge", () => {
    render(<JobProgressView job={job("cancelled")} />);
    expect(screen.getByText("cancelled").className).toContain("ui-badge--warn");
  });
});
