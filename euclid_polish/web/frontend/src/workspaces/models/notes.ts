/* The tracking-notebook note of Models › Universal (pure; notes.test.ts):
   the markdown its "Log to notebook" button (workspaces/shared/LogToNotebook)
   opens with — the universal models table with its definitions; the user
   edits it before appending. */
import { kneeNum, db, dbDelta } from "./model";
import { universalLabel } from "./production";
import { hardwareText, hoursText, kneeWords, stepImplText, stepsShort, type ShapedRow } from "./universal";

const range = (from: number, to: number) => `${kneeNum(from)}–${kneeNum(to)} e⁻`;
const row = (cells: readonly string[]) => `| ${cells.join(" | ")} |`;

export function universalNote(rows: readonly ShapedRow[], opts: {
  nFields: number; integration?: { from_e: number; to_e: number } | null; production?: string | null;
}): string {
  const from = opts.integration?.from_e ?? 0.1, to = opts.integration?.to_e ?? 1e4;
  const production = opts.production ?? null;
  const sorted = [...rows].sort((a, b) => (b.mean ?? -Infinity) - (a.mean ?? -Infinity));
  const lines = [
    `**Universal models** — ${rows.length} model${rows.length === 1 ? "" : "s"}${production ? ` · production: ${universalLabel(production)}` : ""} · ∫PSNR over ${range(from, to)}, mean of the bands, on ${opts.nFields} test fields`, "",
    row(["Model", "Depth", "Steps", "Batch × crops", "Knee", "Step", "Hardware", "Train h", "∫PSNR", "vs production"]),
    row(["---", "---:", "---:", "---", "---", "---", "---", "---:", "---:", "---:"]),
    ...sorted.map((r) => row([
      `#${r.num}${r.name === production ? " (production)" : ""}`, r.depth == null ? "—" : String(r.depth), stepsShort(r.stepsDone),
      `${r.batch_size ?? "—"} × ${r.crops_per_field ?? "—"}`, kneeWords(r), stepImplText(r), hardwareText(r.hardware),
      r.hours == null ? "—" : hoursText(r.hours).replace(/ h$/, ""), `${db(r.mean)}${r.status === "stale" ? " (stale)" : ""}`,
      `${dbDelta(r.vsProduction)}${r.stale && r.vsProduction != null ? " (stale)" : ""}`,
    ])),
    "", "Training time: Σ of the training log's windows (rolled-back ones included), on the hardware named.",
  ];
  return lines.join("\n");
}
