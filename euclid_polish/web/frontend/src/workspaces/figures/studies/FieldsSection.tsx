/* A study's attached fields (≤ 10, stored on holylabs). Nothing is fetched
 * on open: "Fetch field" brings the core products (LR, HR, mean, gate, mask),
 * member SRs come one by one or as "every member that fits" (the rest are
 * named as skipped). One fetch runs at a time across the console, so every
 * fetch button is disabled while one runs. A fetched field opens in the
 * viewer (`study` collection); a member tier is offered only once fetched. */
import { useId, useMemo, useRef, useState } from "react";
import { useJob, useJobsFeed } from "../../../api/jobs";
import { invalidate } from "../../../api/query";
import { useFasrcStatus } from "../../../app/status";
import { formatBytes, formatCount } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { Badge, Button, Caption, Chip, EmptyState, JobProgress, Select } from "../../../ui";
import { ImageViewer, type ViewerApi } from "../../../viewer";
import { studyUrl, type StudyDetail, type StudyField } from "./api";
import { memberName, memberNumber } from "./model";

export const FETCH_JOB_KEY = "study:fetch";
const FETCH_KIND = "study-fetch";

const KIND_WORD: Record<string, string> = { test: "synthetic test field", blackout: "blackout test field", real: "real tile" };
const DEFAULT_TIERS: Record<string, string[]> = { test: ["lr", "sr", "hr"], blackout: ["lr", "sr", "mask"], real: ["lr", "sr"] };

type FetchResult = { study_id?: string; fid?: string; fetched?: string[]; skipped?: string[]; cached?: string[]; bytes?: number };

const productMember = (label: string) => `member_${memberNumber(label)}`;

function sizesText(f: StudyField): string {
  const members = Object.values(f.member_bytes ?? {});
  const total = members.reduce((s, b) => s + (b || 0), 0);
  const core = `core ${formatBytes(f.core_bytes)}`;
  return members.length ? `${core} · ${formatCount(members.length)} member SRs ${formatBytes(total)}` : core;
}

function FieldRow({ f, labels, blocked, blockedId, onFetch, onView, viewing, lastSkipped }: {
  f: StudyField; labels: string[]; blocked: boolean; blockedId: string; viewing: boolean; lastSkipped: string[] | null;
  onFetch: (fid: string, products: string) => void; onView: (fid: string) => void;
}) {
  const ownId = useId();
  const [member, setMember] = useState("");
  const uploaded = f.state === "uploaded";
  const cached = new Set(f.cached_products);
  const missing = labels.filter((l) => f.member_bytes?.[productMember(l)] != null && !cached.has(productMember(l)));
  const pick = missing.some((l) => productMember(l) === member) ? member : missing[0] ? productMember(missing[0]) : "";
  const disabled = !uploaded || blocked;
  // Why the fetch buttons are off, as the visible text that says so.
  const why = !uploaded ? ownId : blocked ? blockedId : undefined;
  return (
    <li className="stu-field" data-viewing={viewing || undefined}>
      {f.thumb_url ? <img className="stu-field__thumb" src={f.thumb_url} alt="" width={88} height={88} loading="lazy" /> : <span className="stu-field__thumb" aria-hidden />}
      <div className="stu-field__main">
        <div className="stu-field__head">
          <span className="stu-field__label">{f.label}</span>
          <span className="muted">{KIND_WORD[f.kind] ?? f.kind}</span>
          {!uploaded && <Badge tone="warn" size="sm">not uploaded</Badge>}
        </div>
        <p className="stu-field__sizes">{sizesText(f)} on holylabs{f.fetched ? ` · fetched, ${formatCount(f.members_fetched)} of ${formatCount(Object.keys(f.member_bytes ?? {}).length)} members here` : ""}</p>
        {!uploaded && <p id={ownId} className="stu-field__skipped">Never uploaded: resume the study to attach it.</p>}
        {lastSkipped && lastSkipped.length > 0 && (
          <p className="stu-field__skipped">Skipped (the 2 GiB field cache or the 5 GiB free-disk margin): {lastSkipped.map((p) => memberName(p.replace(/^member_/, ""))).join(", ")}</p>
        )}
        <div className="stu-field__actions">
          {!f.fetched && <Button size="sm" variant="primary" disabled={disabled} aria-describedby={why} onClick={() => onFetch(f.fid, "core")}>Fetch field</Button>}
          {f.fetched && <Button size="sm" variant={viewing ? "primary" : "default"} onClick={() => onView(f.fid)}>{viewing ? "In the viewer" : "View"}</Button>}
          {missing.length > 0 && (
            <>
              <Select size="sm" aria-label={`Member to fetch for ${f.label}`} value={pick} onChange={setMember} disabled={disabled}
                options={missing.map((l) => ({ value: productMember(l), label: `${memberName(l)} · ${formatBytes(f.member_bytes[productMember(l)])}` }))} />
              <Button size="sm" disabled={disabled || !pick} aria-describedby={why} onClick={() => onFetch(f.fid, pick)}>Fetch member</Button>
              <Button size="sm" disabled={disabled} aria-describedby={why} onClick={() => onFetch(f.fid, "members")}>Fetch every member that fits</Button>
            </>
          )}
        </div>
      </div>
    </li>
  );
}

export function FieldsSection({ detail }: { detail: StudyDetail }) {
  const id = detail.study.id;
  const fields = detail.fields;
  const labels = useMemo(() => (detail.manifest.ensemble?.members ?? []).map((m) => m.label), [detail]);
  const fetchJob = useJob(FETCH_JOB_KEY);
  const feed = useJobsFeed({ slurm: false });
  const other = feed.running.find((j) => j.kind === FETCH_KIND && j.job_id !== fetchJob.job?.job_id) ?? null;
  const fasrc = useFasrcStatus().data;
  const offline = fasrc ? !fasrc.ssh_connected : false;
  // One fetch at a time, and only with FASRC (the fields live on holylabs).
  const blocked = fetchJob.busy ? "A fetch is running: the fetch buttons wait for it to finish."
    : other ? `Another field is being fetched (${other.label}); one fetch runs at a time.`
      : offline ? `FASRC is not connected${fasrc?.last_error ? ` (${fasrc.last_error})` : ""}: fields are fetched from holylabs. Connect in System › Connections.` : null;
  const blockedId = useId();
  const [view, setView] = useUrlState("field", "");
  const shown = fields.find((f) => f.fid === view && f.fetched) ?? null;
  const api = useRef<ViewerApi | null>(null);
  const [tiers, setTiers] = useState<string[]>([]);

  const result = (fetchJob.job?.status === "done" ? fetchJob.job.result : null) as FetchResult | null;
  const resultFor = result?.study_id === id ? result.fid ?? null : null;

  const startFetch = (fid: string, products: string) => {
    void fetchJob.run(`${studyUrl(id)}/fields/${encodeURIComponent(fid)}/fetch`, { products }, {
      onDone: () => invalidate(studyUrl(id)),
    });
  };

  if (!fields.length) {
    return <p className="stu-empty">No fields are attached to this study (it was frozen without fields).</p>;
  }
  const fetchedMembers = shown ? labels.map((l, i) => ({ l, i })).filter(({ l }) => shown.cached_products.includes(productMember(l))) : [];
  return (
    <div className="stu-stack">
      {blocked && <p id={blockedId} className="stu-reason" role="status">{blocked}</p>}
      <ul className="stu-fields" aria-label="Attached fields">
        {fields.map((f) => (
          <FieldRow key={f.fid} f={f} labels={labels} blocked={blocked != null} blockedId={blockedId} viewing={shown?.fid === f.fid} onFetch={startFetch}
            onView={(fid) => setView(fid)} lastSkipped={resultFor === f.fid ? result?.skipped ?? [] : null} />
        ))}
      </ul>
      {(fetchJob.job || fetchJob.error) && <JobProgress job={fetchJob.job} error={fetchJob.error} />}
      <Caption>Fields are stored on holylabs and fetched only when you press a fetch button (they need FASRC); fetched products stay in a 2 GiB local cache.</Caption>
      {shown ? (
        <div className="stu-viewer">
          <ImageViewer key={`${shown.fid}:${shown.cached_products.join(",")}`} collection={shown.viewer.collection} params={shown.viewer.params}
            initialId={shown.viewer.id} tiers={DEFAULT_TIERS[shown.kind] ?? ["lr", "sr"]} nav={false} urlKey="stu" toolbar="full"
            onReady={(v) => { api.current = v; }} onState={(s) => setTiers(s.tiers)} />
          {fetchedMembers.length > 0 && (
            <div className="stu-member-chips" role="group" aria-label="Fetched member SRs">
              <span className="muted">Member SRs:</span>
              {fetchedMembers.map(({ l, i }) => {
                const key = `member${i}`;
                const on = tiers.includes(key);
                return (
                  <Chip key={key} on={on} onClick={() => api.current?.setTiers(on ? tiers.filter((t) => t !== key) : [...tiers, key])}>
                    {memberName(l)}
                  </Chip>
                );
              })}
            </div>
          )}
        </div>
      ) : fields.some((f) => f.fetched) ? (
        <EmptyState compact icon="image" title="Pick a fetched field">Press View on a fetched field to open it in the viewer.</EmptyState>
      ) : null}
    </div>
  );
}
