/* "Recommended from N past runs" (spec 2026-09-30, resource advisor): the
 * compact callout under a submission form's resources — the FASRC step card
 * (every step, Synthetic › generate included) and Models › Train's Resources
 * card. It asks `POST /api/fasrc/resources/<step>/recommend` with the form's
 * task params and resources (debounced; read-only: the advisor reads the local
 * job ledger and mutates nothing, like Train's command preview), lists what it
 * would change (current → recommended, with the evidence), keeps the notes and
 * warnings in a disclosure, and changes nothing until Apply (disabled while
 * an edit is being re-asked: the answer on screen is for the previous plan).
 * Loading, no history and errors are one quiet line: never a big error inside
 * a form.
 *
 *   <ResourceAdvice stepId="ensemble_train" params={params} resources={res}
 *     fields={["n_cpus", "memory", "time_limit"]} perTask="member" onApply={(r) => …} />
 */
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ApiError, apiPost } from "../../api/client";
import { Button, Callout, Details, Icon } from "../../ui";
import {
  ADVICE_FIELDS, adviceHeadline, adviceRows, appliedResources, recommendUrl, usageHref,
  type AdviceField, type AdviceResources, type AdviceRow, type Recommendation,
} from "./resourceAdviceModel";
import "./resourceAdvice.css";

export type ResourceAdviceProps = {
  stepId: string;
  /** The task params the host submits (the same params its submit posts). */
  params: Record<string, unknown>;
  /** The form's current resources (any may be missing). */
  resources: Partial<AdviceResources>;
  /** Apply: the current resources with the recommended values in place. */
  onApply: (resources: AdviceResources) => void;
  /** The resource fields the host edits (default all four); a change to
   *  another one (a fixed CPU or GPU count) is not offered. */
  fields?: readonly AdviceField[];
  /** One array task's word ("member"): the labels read "Memory / member". */
  perTask?: string;
  /** Debounce before asking, ms. */
  delay?: number;
};

type Answer = { key: string; rec: Recommendation | null; error: string | null };

/** The change list: "Memory 32G → 36G" and the evidence under it. */
export function AdviceChanges({ rows }: { rows: readonly AdviceRow[] }) {
  return (
    <ul className="radv__changes">
      {rows.map((r) => (
        <li key={r.field} className="radv__change">
          <span className="radv__what">
            <span className="radv__field">{r.label}</span>
            <span className="mono radv__from">{r.current}</span>
            <span aria-hidden="true">→</span><span className="sr-only">to</span>
            <strong className="mono">{r.recommended}</strong>
          </span>
          {r.reason && <span className="radv__reason">{r.reason}</span>}
        </li>
      ))}
    </ul>
  );
}

/** The warnings, then the notes, as plain lists (a host wraps them). */
export function AdviceNotes({ rec }: { rec: Pick<Recommendation, "notes" | "warnings"> }) {
  return (
    <>
      {rec.warnings.length > 0 && (
        <ul className="radv__notes radv__notes--warn" aria-label="Warnings">
          {rec.warnings.map((w, i) => <li key={i}><Icon name="warn" size={12} /> {w}</li>)}
        </ul>
      )}
      {rec.notes.length > 0 && (
        <ul className="radv__notes" aria-label="Notes">{rec.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
      )}
    </>
  );
}

const notesSummary = (rec: Recommendation): string => {
  const w = rec.warnings.length, n = rec.notes.length;
  return [w ? `${w} warning${w === 1 ? "" : "s"}` : "", n ? `${n} note${n === 1 ? "" : "s"}` : ""].filter(Boolean).join(" · ");
};

export function ResourceAdvice({
  stepId, params, resources, onApply, fields = ADVICE_FIELDS, perTask, delay = 400,
}: ResourceAdviceProps) {
  const key = JSON.stringify([stepId, params, resources]);
  const [answer, setAnswer] = useState<Answer | null>(null);
  useEffect(() => {
    const ctl = new AbortController();
    const t = window.setTimeout(() => {
      apiPost<Recommendation>(recommendUrl(stepId), { params, resources }, { json: true, signal: ctl.signal })
        .then((rec) => {
          if (ctl.signal.aborted) return;
          setAnswer(rec && rec.ok !== false ? { key, rec, error: null } : { key, rec: null, error: rec?.error ?? "refused" });
        })
        .catch((e) => {
          if (ctl.signal.aborted) return;
          // A step the ledger has never seen answers 404: no history, not an error.
          if (e instanceof ApiError && e.status === 404) setAnswer({ key, rec: null, error: null });
          else setAnswer({ key, rec: null, error: e instanceof Error ? e.message : String(e) });
        });
    }, delay);
    return () => { ctl.abort(); window.clearTimeout(t); };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps

  const rec = answer?.rec ?? null;
  const rows = useMemo(() => adviceRows(rec, fields, perTask), [rec, fields, perTask]);
  const stale = !!answer && answer.key !== key;

  if (!answer) return <p className="radv__quiet" role="status">Checking past runs…</p>;
  if (answer.error) {
    return <p className="radv__quiet">Past-run advice is unavailable ({answer.error}).</p>;
  }
  if (!rec || !rec.available) {
    return <p className="radv__quiet">No finished run of this step to recommend resources from yet.</p>;
  }
  const extra = notesSummary(rec);
  return (
    <Callout tone="info" className="radv" title={adviceHeadline(rec)}>
      <div className="radv__body" data-stale={stale || undefined}>
        {rows.length ? <AdviceChanges rows={rows} /> : <span>The form already asks for this.</span>}
        {extra && (
          <Details summary={extra} className="radv__details">
            <AdviceNotes rec={rec} />
          </Details>
        )}
        <div className="radv__actions">
          <Button size="sm" variant="primary" disabled={!rows.length || stale}
            title={stale ? "Checking past runs for the edited plan…" : undefined}
            onClick={() => onApply(appliedResources(resources, rec, fields))}>Apply</Button>
          <Button asChild size="sm" variant="ghost" iconRight="chevronRight">
            <Link to={usageHref(stepId)}>Open usage</Link>
          </Button>
        </div>
      </div>
    </Callout>
  );
}
