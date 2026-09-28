/* One server-rendered calibration plate (galaxy population, galaxy
 * distributions, stellar population): a preview, its caption line, the
 * export resolution and PNG / PDF / SVG downloads, and the tab its data is
 * judged on. The plates render the reviewed cached calibration only —
 * opening one never fits or activates anything. */
import { useState } from "react";
import { Link } from "react-router-dom";
import { Button, Select, Switch, Tooltip, type SelectOption } from "../../../ui";
import { PLATE_DPIS } from "../api";
import { ServerImage } from "../common";
import { PlateHead } from "./PlateHead";
import type { PlateCaption } from "./plateStatus";

export type StaticPlateDef = {
  id: string;
  title: string;
  sub: string;
  /** The plate URL (`format`, `dpi`, `inline`, extra params). */
  url: (opts: { format: "png" | "pdf" | "svg"; dpi: number; inline?: boolean; training?: boolean }) => string;
  /** Where the underlying data is reviewed / refreshed. */
  source: { to: string; label: string };
  training?: boolean;
  minHeight: number;
};

export const DPI_OPTIONS: SelectOption[] = PLATE_DPIS.map((d) => ({ value: String(d), label: `${d} dpi` }));

export function StaticPlate({ plate, caption, dpi, onDpi, training, onTraining }: {
  plate: StaticPlateDef; caption: PlateCaption; dpi: number; onDpi: (dpi: string) => void;
  training: boolean; onTraining: (on: boolean) => void;
}) {
  const withTraining = plate.training ? training : undefined;
  const preview = plate.url({ format: "png", dpi: 150, inline: true, training: withTraining });
  // A plate the server cannot render (no reviewed artifact yet) has nothing to download.
  const [failed, setFailed] = useState<string | null>(null);
  const off = failed === preview;
  const dl = (format: "png" | "pdf" | "svg") => (off ? undefined : plate.url({ format, dpi, training: withTraining }));
  return (
    <section className="fig-plate" aria-labelledby={`fig-plate-${plate.id}`}>
      <PlateHead id={plate.id} title={plate.title} sub={plate.sub} caption={caption} tools={<>
        {plate.training && (
          <Tooltip content="Draw the training-set population too (only the generated curves change)">
            <Switch size="sm" checked={training} onChange={onTraining}>Training</Switch>
          </Tooltip>
        )}
        <Select size="sm" value={String(dpi)} onChange={onDpi} options={DPI_OPTIONS} aria-label="Download resolution" />
        <Button size="sm" icon="download" disabled={off} href={dl("png")} download>PNG</Button>
        <Button size="sm" disabled={off} href={dl("pdf")} download>PDF</Button>
        <Button size="sm" disabled={off} href={dl("svg")} download>SVG</Button>
        <Button asChild size="sm" variant="ghost" iconRight="chevronRight"><Link to={plate.source.to}>{plate.source.label}</Link></Button>
      </>} />
      <ServerImage src={preview} alt={plate.title} minHeight={plate.minHeight} className="fig-plate__image"
        onError={() => setFailed(preview)} />
    </section>
  );
}
