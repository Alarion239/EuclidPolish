/* ensemble/disagreement (spec §8.2): the viewer v2 over the `ensemble`
   collection — `sr` is the production spatial gate, `mean` the plain mean —
   and a member picker driving it: one member shows its SR as a still, two or
   more play the disagreement movie decomposed over just those members. The
   picked members are in the URL (?sel=196,195; the Members tab links here). */
import { useCallback, useEffect, useMemo, useRef } from "react";
import { useResource } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import { useUrlState } from "../../../hooks/useUrlState";
import { Button, Chip, EmptyState, Page, Select } from "../../../ui";
import { ImageViewer, type ViewerApi } from "../../../viewer";
import { url, useMembers, useMode, type MemberRow } from "../api";
import { BarGroup, EnsBar, useFacetColors } from "../common";
import { db, kneeText, memberNumber } from "../model";
import "../ensemble.css";

type Sort = "index" | "knee" | "psnr" | "loss";
type Meta = { member_labels?: string[]; count?: number };
type Pick = { i: number; num: string; label: string; row: MemberRow | null };

export default function Disagreement() {
  const mode = useMode();
  const meta = useResource<Meta>(url.viewerMeta(mode), [mode], { ttl: 0 });
  const members = useMembers(mode);
  const [selRaw, setSelRaw] = useUrlState("sel", "");
  const [sort, setSort] = useUrlState<Sort>("sort", "knee");
  const [loss, setLoss] = useUrlState("loss", "");
  const api = useRef<ViewerApi | null>(null);
  /** The viewer takes tier changes once its meta has loaded (first state with tiers). */
  const ready = useRef(false);

  const byNum = useMemo(() => new Map((members.data?.members ?? []).map((m) => [memberNumber(m.name) ?? "", m])), [members.data]);
  const picks = useMemo<Pick[]>(() => (meta.data?.member_labels ?? []).map((label, i) => {
    const num = memberNumber(label) ?? String(i);
    return { i, num, label, row: byNum.get(num) ?? null };
  }), [meta.data, byNum]);
  const colors = useFacetColors(picks.map((p) => p.row ?? { loss: "l1" }), "loss");
  const losses = useMemo(() => [...new Set(picks.map((p) => p.row?.loss ?? "l1"))].sort(), [picks]);
  const sel = useMemo(() => selRaw.split(",").map((s) => memberNumber(s)).filter((n): n is string => !!n), [selRaw]);
  const selSet = new Set(sel);
  const shown = useMemo(() => {
    const f = picks.filter((p) => !loss || (p.row?.loss ?? "l1") === loss);
    const cmp: Record<Sort, (a: Pick, b: Pick) => number> = {
      index: (a, b) => a.i - b.i,
      knee: (a, b) => (b.row?.knee_integrated?.mean ?? -1e9) - (a.row?.knee_integrated?.mean ?? -1e9),
      psnr: (a, b) => (b.row?.psnr ?? -1e9) - (a.row?.psnr ?? -1e9),
      loss: (a, b) => (a.row?.loss ?? "").localeCompare(b.row?.loss ?? "") || a.i - b.i,
    };
    return [...f].sort(cmp[sort]);
  }, [picks, loss, sort]);

  /** Map the selection onto the viewer, keeping the user's base tiers. */
  const apply = useCallback((nums: string[]) => {
    const v = api.current;
    if (!v) return;
    const idx = nums.map((n) => picks.find((p) => p.num === n)?.i).filter((i): i is number => i != null).sort((a, b) => a - b);
    const base = v.getState().tiers.filter((t) => !/^member\d+$/.test(t) && t !== "morph");
    if (idx.length >= 2) {
      v.setMorphMembers(idx.join(","));
      v.setTiers([...base, "morph"]);
    } else if (idx.length === 1) {
      v.setMorphMembers(null);
      v.setTiers([...base, `member${idx[0]}`]);
    } else {
      v.setMorphMembers(null);
      v.setTiers(base.length ? base : ["sr"]);
    }
  }, [picks]);
  useEffect(() => { if (ready.current) apply(sel); }, [apply, sel.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = (num: string) => setSelRaw((selSet.has(num) ? sel.filter((n) => n !== num) : [...sel, num]).join(","));
  const top = (k: number) => setSelRaw([...picks].filter((p) => p.row?.knee_integrated?.mean != null)
    .sort((a, b) => (b.row!.knee_integrated!.mean as number) - (a.row!.knee_integrated!.mean as number)).slice(0, k).map((p) => p.num).join(","));
  usePageActions([
    { id: "dis-top5", label: "Disagreement movie over the top 5 members (∫PSNR)", group: "Disagreement", run: () => top(5) },
    { id: "dis-clear", label: "Disagreement: clear the member selection", group: "Disagreement", disabled: !sel.length, run: () => setSelRaw("") },
    { id: "dis-reload", label: "Disagreement: reload the cube cache", group: "Disagreement", run: () => { void meta.reload(); api.current?.reload(); } },
  ]);

  const status = sel.length >= 2 ? `movie over ${sel.length} members` : sel.length === 1 ? `member #${sel[0]} (still)` : "pick members: one shows its SR, two or more play the movie";
  const count = meta.data?.count ?? 0;
  return (
    <Page>
      <EnsBar label="Disagreement controls">
        <span className="ens-muted">{status}</span>
        <span className="ens-bar__spacer" />
        <Button size="sm" onClick={() => top(5)} disabled={!picks.some((p) => p.row?.knee_integrated)}>top 5</Button>
        <Button size="sm" variant="ghost" disabled={!sel.length} onClick={() => setSelRaw("")}>clear</Button>
        <Button size="sm" variant="ghost" icon="reset" aria-label="Reload the cube cache"
          onClick={() => { void meta.reload(); api.current?.reload(); }} />
      </EnsBar>
      {meta.error ? (
        <EmptyState icon="warn" title="The ensemble cube cache is not readable"><span className="ens-mono">{meta.error.message}</span></EmptyState>
      ) : !meta.loading && count === 0 ? (
        <EmptyState icon="image" title={`No ${mode} test cubes cached`}>Evaluate the ensemble (Overview) to cache the test fields for the viewer.</EmptyState>
      ) : (
        <div className="ens-stack">
          <ImageViewer key={mode} collection="ensemble" params={{ mode }} urlKey="ens" toolbar="full"
            onReady={(v) => { api.current = v; if (!v) ready.current = false; }}
            onState={(s) => {
              if (!ready.current && s.tiers.length && picks.length) { ready.current = true; apply(sel); }
            }} />
          <div className="ens-row">
            <BarGroup label="Members">
              {losses.length > 1 && losses.map((l) => (
                <Chip key={l} on={loss === l} onClick={() => setLoss(loss === l ? "" : l)}>{l}</Chip>
              ))}
              <Select<Sort> size="sm" aria-label="Sort members" value={sort} onChange={setSort}
                options={[{ value: "knee", label: "by ∫PSNR" }, { value: "psnr", label: "by test PSNR" }, { value: "loss", label: "by loss" }, { value: "index", label: "by index" }]} />
            </BarGroup>
          </div>
          <div className="ens-picker" role="group" aria-label="Members in the movie">
            {shown.map((p) => (
              <button key={p.i} type="button" className="ens-pick" data-on={selSet.has(p.num)} aria-pressed={selSet.has(p.num)}
                style={{ ["--sw" as string]: colors.of(p.row ?? { loss: "l1" }) }} onClick={() => toggle(p.num)}
                title={p.row ? kneeText(p.row).title : p.label}>
                <span className="ens-pick__top"><span>#{p.num}</span><span>{db(p.row?.knee_integrated?.mean)}</span></span>
                <span className="ens-pick__meta">{p.row?.loss ?? "?"} · {p.row ? kneeText(p.row).text : ""}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </Page>
  );
}
