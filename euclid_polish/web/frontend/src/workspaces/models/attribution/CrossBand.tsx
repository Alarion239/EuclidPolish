/* Models › Knees › Cross-band: a 4×4 heat table per model — the mean share
   of an output band's sensitivity (row) that flows through each input band
   (column), over the sampled pixels of the chosen set. The cell tint grows
   with the share (an accent mix, so the text stays readable in both themes). */
import type { KneeAggregate } from "../kneesApi";
import { BAND_LABEL, KNEE_BANDS, crossBandMatrix, sharePct } from "../knees";

export function CrossBand({ agg, title }: { agg: KneeAggregate | null | undefined; title: string }) {
  const m = crossBandMatrix(agg);
  if (!agg || m.every((row) => row.every((v) => v == null))) return null;
  return (
    <figure className="kn-crossfig">
      <figcaption className="mdl-chart__title">{title}</figcaption>
      <table className="kn-cross" aria-label={`Cross-band shares, ${title}`}>
        <thead>
          <tr>
            <th scope="col"><span className="mdl-faint">out ↓ in →</span></th>
            {KNEE_BANDS.map((b) => <th key={b} scope="col">{BAND_LABEL[b]}</th>)}
          </tr>
        </thead>
        <tbody>
          {KNEE_BANDS.map((b, r) => (
            <tr key={b}>
              <th scope="row">{BAND_LABEL[b]}</th>
              {m[r].map((v, c) => (
                <td key={c} data-diag={r === c || undefined}
                  style={v == null ? undefined : { ["--mix" as string]: `${Math.round(Math.max(0, Math.min(1, v)) * 60)}%` }}>
                  {sharePct(v)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}
