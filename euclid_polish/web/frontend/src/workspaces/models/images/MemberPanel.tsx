/* Models › Images, the ONE member picker (a side panel of the page, beside
 * the viewer on a wide page and under it on a narrow one; the tab strip
 * carries no popover, so it never reshapes). What the viewer shows is its
 * status line; Top 5 by ∫PSNR and Clear; a search box (number, loss, knee),
 * loss chips, the sort and the member toggles. One member adds its SR still,
 * two or more play the disagreement movie over just those members. The
 * ranking itself is the Leaderboard's. */
import { Link } from "react-router-dom";
import { Button, Chip, Input, Select } from "../../../ui";
import type { MemberRow } from "../api";
import { db, kneeText } from "../model";

export type Sort = "index" | "knee" | "psnr" | "loss";
export type Pick = { i: number; num: string; label: string; row: MemberRow | null };

const SORTS: { value: Sort; label: string }[] = [
  { value: "knee", label: "By ∫PSNR" }, { value: "psnr", label: "By test PSNR" },
  { value: "loss", label: "By loss" }, { value: "index", label: "By number" },
];

export function MemberPanel({ shown, picks, sel, status, find, setFind, loss, setLoss, losses, sort, setSort, colorOf, toggle, top, clear, ranked, loading = false, unreadable = false, leaderboard }: {
  shown: Pick[]; picks: Pick[]; sel: Set<string>; status: string; find: string; setFind: (v: string) => void;
  loss: string; setLoss: (v: string) => void; losses: string[]; sort: Sort; setSort: (v: Sort) => void;
  colorOf: (row: { loss: string }) => string; toggle: (num: string) => void; top: (k: number) => void; clear: () => void;
  ranked: boolean; leaderboard: string;
  /** The members table: still loading (the cards give no verdict and Top 5
   *  waits), unreadable, or read. */
  loading?: boolean; unreadable?: boolean;
}) {
  return (
    <aside className="mdl-members" aria-labelledby="mdl-img-members">
      <header className="mdl-members__head">
        <h2 id="mdl-img-members" className="mdl-members__title">Members</h2>
        <span className="mdl-members__status" role="status" aria-live="polite">{status}</span>
      </header>
      <div className="mdl-row">
        <Button size="sm" onClick={() => top(5)} disabled={!ranked || loading} loading={loading && !ranked}
          title={loading ? "Reading the members table…" : ranked ? "The five members with the highest knee-integrated PSNR" : "No knee-integrated PSNR yet (Leaderboard › Knee PSNR)"}>Top 5 by ∫PSNR</Button>
        <Button size="sm" variant="ghost" disabled={!sel.size} onClick={clear}>Clear</Button>
      </div>
      <div className="mdl-members__tools">
        <Input size="sm" type="search" icon="search" clearable value={find} onChange={setFind}
          placeholder="Find: 196, L2, multi" aria-label="Find members" className="mdl-members__find" />
        <Select<Sort> size="sm" aria-label="Sort members" value={sort} onChange={setSort} options={SORTS} />
      </div>
      {losses.length > 1 && (
        <span className="mdl-members__losses" role="group" aria-label="Show one loss">
          {losses.map((l) => (
            <Chip key={l} on={loss === l} dot={colorOf({ loss: l })} onClick={() => setLoss(loss === l ? "" : l)}
              title={loss === l ? "Show every loss" : `Show only ${l.toUpperCase()} members`}>{l.toUpperCase()}</Chip>
          ))}
        </span>
      )}
      <div className="mdl-picker mdl-picker--dense mdl-picker--side" role="group" aria-label="Members in the movie">
        {shown.map((p) => {
          const k = p.row ? kneeText(p.row) : null;
          const on = sel.has(p.num);
          return (
            <button key={p.i} type="button" className="mdl-pick" data-on={on} aria-pressed={on}
              style={{ ["--sw" as string]: colorOf(p.row ?? { loss: "l1" }) }} onClick={() => toggle(p.num)}
              title={`${p.row ? `${p.row.loss.toUpperCase()} · ${k?.title}` : p.label}${p.row?.knee_integrated?.mean != null ? ` · ∫PSNR ${db(p.row.knee_integrated.mean)} dB` : ""}`}>
              <span className="mdl-pick__top"><span>#{p.num}</span><span>{db(p.row?.knee_integrated?.mean)}</span></span>
              <span className="mdl-pick__meta">{p.row ? `${p.row.loss.toUpperCase()} · ${k?.text}` : loading ? "…" : unreadable ? "members table not readable" : "not in the members table"}</span>
            </button>
          );
        })}
        {!shown.length && picks.length > 0 && <p className="mdl-faint mdl-picker__none">No member matches{find ? ` “${find}”` : " this loss"}.</p>}
      </div>
      <p className="mdl-faint">∫PSNR in dB; the full ranking is the <Link to={leaderboard}>Leaderboard</Link>.</p>
    </aside>
  );
}
