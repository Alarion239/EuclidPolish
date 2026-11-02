/* A battery study (`/figures/studies?study=<id>`, kind "battery"): chosen
 * single models compared by the model battery (scripts/model_battery.py
 * --study), frozen as they were measured. The comparison table and the
 * paired comparisons come from the frozen battery.json, the figures and the
 * report are frozen files; nothing is recomputed. Reads only on open. */
import { Fragment } from "react";
import { Link, useNavigate } from "react-router-dom";
import { formatDate, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { Button, Callout, Details, FactsList, Num, PngFigure, Section, SummaryLine, Table, Tabs, buttonClass, type Column } from "../../../ui";
import { deleteStudy } from "./actions";
import { studyUrl, type BatteryModel, type BatteryRow, type StudyDetail } from "./api";
import { NoteEditor } from "./NoteEditor";

const LIST_PATH = "/figures/studies";
const VIEWS = ["table", "paired", "figures"] as const;
type View = (typeof VIEWS)[number];
const VIEW_LABEL: Record<View, string> = { table: "Results", paired: "Paired vs baseline", figures: "Figures" };

/** `figure-curve_of_growth.png` → "Curve of growth". */
export function figureTitle(number: string): string {
  const stem = number.replace(/^figure-/, "").replace(/\.png$/, "").replace(/_/g, " ");
  return stem.charAt(0).toUpperCase() + stem.slice(1);
}

/** The rows grouped by their section, in order. */
export function sections(rows: readonly BatteryRow[]): { title: string; rows: BatteryRow[] }[] {
  const out: { title: string; rows: BatteryRow[] }[] = [];
  for (const row of rows) {
    const last = out[out.length - 1];
    if (last && last.title === row[0]) last.rows.push(row);
    else out.push({ title: row[0], rows: [row] });
  }
  return out;
}

function ResultTable({ title, rows, models }: { title: string; rows: BatteryRow[]; models: readonly BatteryModel[] }) {
  const columns: Column<BatteryRow>[] = [
    { header: "Metric", cell: (r) => r[1] },
    ...models.map((m, j) => ({ header: m.name, align: "right" as const, cell: (r: BatteryRow) => <span className="stu-tnum">{r[2][j] ?? "—"}</span> })),
  ];
  return (
    <div className="stu-panel">
      <h3 className="stu-panel__title">{title}</h3>
      <div className="stu-table"><Table columns={columns} rows={rows} aria-label={title} /></div>
    </div>
  );
}

const fileUrl = (id: string, name: string, download = false) => `${studyUrl(id)}/numbers/${encodeURIComponent(name)}${download ? "?download=1" : ""}`;

export function BatteryStudy({ detail, reload }: { detail: StudyDetail; reload: () => void }) {
  const m = detail.manifest;
  const battery = m.battery ?? { models: [], tests: [], figures: [] };
  const models = battery.models;
  const numbers = detail.numbers?.battery;
  const navigate = useNavigate();
  const [viewRaw, setView] = useUrlState("view", "table");
  const view: View = (VIEWS as readonly string[]).includes(viewRaw) ? (viewRaw as View) : "table";
  const [figureRaw, setFigure] = useUrlState("figure", "");
  const figure = battery.figures.includes(figureRaw) ? figureRaw : battery.figures[0] ?? "";
  const id = detail.study.id;
  const complete = detail.study.state === "complete";
  const records = battery.records_dir ? battery.records_dir.split("/").filter(Boolean).slice(-1)[0] : null;

  return (
    <div className="stu-stack">
      <div className="stu-head">
        <Button size="sm" variant="ghost" icon="chevronLeft" asChild><Link to={LIST_PATH}>All studies</Link></Button>
        <h2 className="stu-title">{detail.study.name}</h2>
        <span className="stu-head__right">
          <Button size="sm" variant="ghost" onClick={() => void deleteStudy(detail.study).then((gone) => { if (gone) navigate(LIST_PATH); })}>Delete study…</Button>
        </span>
      </div>
      <SummaryLine>
        Battery comparison of <Num>{models.length}</Num> models ({models.map((x) => x.name).join(", ")}; the first is the baseline), frozen{" "}
        <Num>{formatRelative(m.completed ?? m.created)}</Num>.
      </SummaryLine>
      {(m.warnings?.length ?? 0) > 0 && (
        <Callout tone="warn" title="Frozen with missing figures"><ul className="stu-warnings">{m.warnings!.map((w) => <li key={w}>{w}</li>)}</ul></Callout>
      )}
      {!complete && <Callout tone="warn" title="This study is incomplete">{detail.study.reason ?? "It was never sealed."}</Callout>}
      <FactsList title="Frozen" facts={[
        { label: "Created", value: formatDate(m.created) },
        { label: "Battery run", value: battery.created ?? "—", hint: battery.seconds ? `${Math.round(battery.seconds / 60)} min` : undefined },
        { label: "Test records", value: records ?? "console mirror", hint: battery.records_dir ?? undefined },
        { label: "Tests", value: battery.tests.filter((t) => t !== "figures").length, hint: battery.tests.join(", ") },
      ]} />
      <NoteEditor detail={detail} onSaved={reload} />
      <Details summary="Provenance">
        <dl className="stu-prov">
          <dt>Study id</dt><dd className="mono">{id}</dd>
          <dt>Manifest sha256</dt><dd className="mono">{detail.manifest_sha256}</dd>
          <dt>Citation</dt><dd>{detail.citation}</dd>
          {battery.records_fp && <><dt>Test records</dt><dd className="mono">{battery.records_fp}</dd></>}
          {models.map((x) => <Fragment key={x.spec}><dt>{x.name}</dt><dd className="mono">{x.spec} · {x.fingerprint}</dd></Fragment>)}
          {battery.source && <><dt>Battery output</dt><dd className="mono">{battery.source}</dd></>}
        </dl>
      </Details>
      <div className="stu-row-actions">
        <a className={buttonClass("default", "sm")} href={fileUrl(id, "battery.md", true)} download>Report (battery.md)</a>
        <a className={buttonClass("default", "sm")} href={fileUrl(id, "battery.json", true)} download>Numbers (battery.json)</a>
      </div>

      {complete && numbers && (
        <Tabs<View> aria-label="Battery" value={view} onChange={setView} variant="line" tabs={VIEWS.map((v) => ({ id: v, label: VIEW_LABEL[v] }))}>
          <div className="stu-chart">
            {view === "table" && sections(numbers.table).map((s) => <ResultTable key={s.title} title={s.title} rows={s.rows} models={models} />)}
            {view === "paired" && (
              <>
                <p className="stu-empty">Each model minus the baseline ({models[0]?.name ?? "—"}), with its confidence interval.</p>
                {sections(numbers.paired).map((s) => <ResultTable key={s.title} title={s.title} rows={s.rows} models={models.slice(1)} />)}
              </>
            )}
            {view === "figures" && (battery.figures.length ? (
              <Section title={figureTitle(figure)} right={<a className={buttonClass("default", "sm")} href={fileUrl(id, figure, true)} download>Download PNG</a>}>
                <PngFigure srcFor={(key) => fileUrl(id, key ?? figure)} active={figure} onActive={setFigure} alt={figureTitle(figure)}
                  toolbar={battery.figures.map((f) => ({ key: f, label: figureTitle(f) }))} minHeight={320} />
              </Section>
            ) : <p className="stu-empty">This battery run froze no figures.</p>)}
          </div>
        </Tabs>
      )}
    </div>
  );
}
