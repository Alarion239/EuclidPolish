/* ensemble/disagreement (spec §8.2; image-first pass 2026-09-27): the viewer
   v2 over the `ensemble` collection — `sr` is the production spatial gate,
   `mean` the plain mean — FIRST, at the top of the page (nothing above its
   bar but the tab strip), sized to the viewport by the viewer's fit.
   It opens on LR | SR | HR (SR = the production gate), so the frame always
   has something to compare against. The members are reachable without
   scrolling from the member menu in the tab strip (right of the tabs,
   ../aside.ts), whose button names what the movie shows ("Members: 196 ·
   Change", "Members: none · Pick"): what is shown, Top 5 /
   Clear, a search box (number, loss, knee), loss filter chips, the sort and
   the member toggles, over the movie so it updates as you pick. The same
   picker sits under the viewer as a panel for bulk work (it shares the
   search, filter and sort). One member shows its SR as a still, two or more
   play the disagreement movie decomposed over just those members. The
   picked members are in the URL (?sel=196,195; the Members tab links here),
   the sort and loss filter too (?sort=, ?loss=). */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useResource } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import { useUrlState } from "../../../hooks/useUrlState";
import { Button, Chip, EmptyState, IconButton, Input, Page, Popover, Select } from "../../../ui";
import { ImageViewer, type ViewerApi } from "../../../viewer";
import { useTabAside } from "../aside";
import { url, useMembers, useMode, type MemberRow } from "../api";
import { useFacetColors } from "../common";
import { db, kneeText, memberMatches, memberNumber, membersButtonText, movieStatus } from "../model";
import "../ensemble.css";

type Sort = "index" | "knee" | "psnr" | "loss";
const DEFAULT_TIERS = ["lr", "sr", "hr"];
type Meta = { member_labels?: string[]; count?: number };
type Pick = { i: number; num: string; label: string; row: MemberRow | null };

const SORTS: { value: Sort; label: string }[] = [
  { value: "knee", label: "By ∫PSNR" }, { value: "psnr", label: "By test PSNR" },
  { value: "loss", label: "By loss" }, { value: "index", label: "By number" },
];

/** The picker's search, loss chips, sort and member toggles — once in the
 *  tab-strip menu, once in the panel under the viewer (shared state). */
function MemberPicker({ shown, picks, sel, find, setFind, loss, setLoss, losses, sort, setSort, colorOf, toggle, variant }: {
  shown: Pick[]; picks: Pick[]; sel: Set<string>; find: string; setFind: (v: string) => void;
  loss: string; setLoss: (v: string) => void; losses: string[]; sort: Sort; setSort: (v: Sort) => void;
  colorOf: (row: { loss: string }) => string; toggle: (num: string) => void; variant: "menu" | "panel";
}) {
  return (
    <>
      <div className="ens-members__tools">
        <Input size="sm" type="search" icon="search" clearable value={find} onChange={setFind}
          placeholder="Find: 196, L2, multi" aria-label="Find members" className="ens-members__find" />
        {losses.length > 1 && (
          <span className="ens-members__losses" role="group" aria-label="Show one loss">
            {losses.map((l) => (
              <Chip key={l} on={loss === l} dot={colorOf({ loss: l })} onClick={() => setLoss(loss === l ? "" : l)}
                title={loss === l ? "Show every loss" : `Show only ${l.toUpperCase()} members`}>{l.toUpperCase()}</Chip>
            ))}
          </span>
        )}
        <Select<Sort> size="sm" aria-label="Sort members" value={sort} onChange={setSort} options={SORTS} />
        <span className="ens-faint ens-members__unit">∫PSNR in dB</span>
      </div>
      <div className={`ens-picker ens-picker--dense ens-picker--${variant}`} role="group" aria-label="Members in the movie">
        {shown.map((p) => {
          const k = p.row ? kneeText(p.row) : null;
          const on = sel.has(p.num);
          return (
            <button key={p.i} type="button" className="ens-pick" data-on={on} aria-pressed={on}
              style={{ ["--sw" as string]: colorOf(p.row ?? { loss: "l1" }) }} onClick={() => toggle(p.num)}
              title={`${p.row ? `${p.row.loss.toUpperCase()} · ${k?.title}` : p.label}${p.row?.knee_integrated?.mean != null ? ` · ∫PSNR ${db(p.row.knee_integrated.mean)} dB` : ""}`}>
              <span className="ens-pick__top"><span>#{p.num}</span><span>{db(p.row?.knee_integrated?.mean)}</span></span>
              <span className="ens-pick__meta">{p.row ? `${p.row.loss.toUpperCase()} · ${k?.text}` : "not in members.json"}</span>
            </button>
          );
        })}
        {!shown.length && picks.length > 0 && <p className="ens-faint ens-picker__none">No member matches{find ? ` “${find}”` : " this loss"}.</p>}
      </div>
    </>
  );
}

export default function Disagreement() {
  const mode = useMode();
  const meta = useResource<Meta>(url.viewerMeta(mode), [mode], { ttl: 0 });
  const members = useMembers(mode);
  const [selRaw, setSelRaw] = useUrlState("sel", "");
  const [sort, setSort] = useUrlState<Sort>("sort", "knee");
  const [loss, setLoss] = useUrlState("loss", "");
  const [find, setFind] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const aside = useTabAside();
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
    const f = picks.filter((p) => (!loss || (p.row?.loss ?? "l1") === loss)
      && memberMatches({ num: p.num, loss: p.row?.loss, knee: p.row ? kneeText(p.row).text : null, label: p.label }, find));
    const cmp: Record<Sort, (a: Pick, b: Pick) => number> = {
      index: (a, b) => a.i - b.i,
      knee: (a, b) => (b.row?.knee_integrated?.mean ?? -1e9) - (a.row?.knee_integrated?.mean ?? -1e9),
      psnr: (a, b) => (b.row?.psnr ?? -1e9) - (a.row?.psnr ?? -1e9),
      loss: (a, b) => (a.row?.loss ?? "").localeCompare(b.row?.loss ?? "") || a.i - b.i,
    };
    return [...f].sort(cmp[sort] ?? cmp.knee);
  }, [picks, loss, sort, find]);

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
  const ranked = picks.some((p) => p.row?.knee_integrated?.mean != null);
  const top = (k: number) => setSelRaw([...picks].filter((p) => p.row?.knee_integrated?.mean != null)
    .sort((a, b) => (b.row!.knee_integrated!.mean as number) - (a.row!.knee_integrated!.mean as number)).slice(0, k).map((p) => p.num).join(","));
  const reload = () => { void meta.reload(); void api.current?.reload(); };
  usePageActions([
    { id: "dis-pick", label: "Disagreement: pick members…", group: "Disagreement", disabled: !aside || !picks.length, run: () => setMenuOpen(true) },
    { id: "dis-top5", label: "Disagreement movie over the top 5 members (∫PSNR)", group: "Disagreement", run: () => top(5) },
    { id: "dis-clear", label: "Disagreement: clear the member selection", group: "Disagreement", disabled: !sel.length, run: () => setSelRaw("") },
    { id: "dis-reload", label: "Disagreement: reload the cube cache", group: "Disagreement", run: reload },
  ]);

  const count = meta.data?.count ?? 0;
  if (meta.error) {
    return (
      <Page>
        <EmptyState icon="warn" title="The ensemble cube cache is not readable"
          action={<Button size="sm" icon="reset" onClick={reload}>Retry</Button>}>
          <span className="ens-mono">{meta.error.message}</span>
        </EmptyState>
      </Page>
    );
  }
  if (!meta.loading && count === 0) {
    return (
      <Page>
        <EmptyState icon="image" title={`No ${mode} test cubes cached`}>Evaluate the ensemble (Overview) to cache the test fields for the viewer.</EmptyState>
      </Page>
    );
  }
  const status = movieStatus(sel);
  const actions = (
    <>
      <Button size="sm" onClick={() => top(5)} disabled={!ranked}
        title={ranked ? "The five members with the highest knee-integrated PSNR" : "No knee-integrated PSNR yet (Knee PSNR tab)"}>Top 5 by ∫PSNR</Button>
      <Button size="sm" variant="ghost" disabled={!sel.length} onClick={() => setSelRaw("")}>Clear selection</Button>
    </>
  );
  const pickerProps = {
    shown, picks, sel: selSet, find, setFind, loss, setLoss, losses, sort, setSort,
    colorOf: (row: { loss: string }) => colors.of(row), toggle,
  };
  const [menuWhat, menuAction] = membersButtonText(sel).split(" · ");
  const menu = aside && picks.length > 0 && createPortal(
    <Popover open={menuOpen} onOpenChange={setMenuOpen} align="end" label="Pick members" className="ens-menu"
      trigger={
        <Button size="sm" variant={sel.length ? "default" : "primary"} iconRight="chevronDown" className="ens-menu__trigger"
          title={`${status}. SR is the production gate; picked members add their own SR frame (one) or the disagreement movie (two or more).`}>
          {menuWhat}<span className="ens-menu__action"> · {menuAction}</span>
        </Button>
      }>
      <div className="ens-menu__head">
        <span className="ens-members__status">{status}</span>
        <span className="ens-bar__spacer" />
        {actions}
      </div>
      <MemberPicker {...pickerProps} variant="menu" />
    </Popover>,
    aside,
  );
  return (
    <Page className="ens-viewer-page">
      {menu}
      {/* LR | SR | HR by default: the SR frame alone had nothing to compare against. */}
      <ImageViewer key={mode} collection="ensemble" params={{ mode }} urlKey="ens" toolbar="full" tiers={DEFAULT_TIERS}
        onReady={(v) => { api.current = v; if (!v) ready.current = false; }}
        onState={(s) => {
          if (!ready.current && s.tiers.length && picks.length) { ready.current = true; apply(sel); }
        }} />
      <section className="ens-members" aria-labelledby="ens-dis-members">
        <header className="ens-members__head">
          <h2 id="ens-dis-members" className="ens-members__title">Members</h2>
          <span className="ens-members__status" role="status" aria-live="polite">{status}</span>
          <span className="ens-bar__spacer" />
          {actions}
          <IconButton size="sm" icon="reset" label="Reload the cube cache" onClick={reload} />
        </header>
        <MemberPicker {...pickerProps} variant="panel" />
      </section>
    </Page>
  );
}
