/* One server-rendered publication plate (population atlas, stellar
 * calibration, galaxy 2×2): a preview and PNG / PDF / SVG downloads. The
 * plates render the reviewed cached artifacts only — opening one never fits
 * or activates a calibration. */
import { useState } from "react";
import { Link } from "react-router-dom";
import { Button, Switch, Tooltip } from "../../../ui";
import { ServerImage } from "../common";

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

export function StaticPlate({ plate, dpi, training, onTraining }: {
  plate: StaticPlateDef; dpi: number; training: boolean; onTraining: (on: boolean) => void;
}) {
  const withTraining = plate.training ? training : undefined;
  const preview = plate.url({ format: "png", dpi: 150, inline: true, training: withTraining });
  // A plate the server cannot render (no reviewed artifact yet) has nothing to download.
  const [failed, setFailed] = useState<string | null>(null);
  const off = failed === preview;
  const dl = (format: "png" | "pdf" | "svg") => (off ? undefined : plate.url({ format, dpi, training: withTraining }));
  return (
    <section className="fig-plate" aria-labelledby={`fig-plate-${plate.id}`}>
      <header className="fig-plate__head">
        <div className="fig-plate__heading">
          <h2 id={`fig-plate-${plate.id}`}>{plate.title}</h2>
          <p className="muted">{plate.sub}</p>
        </div>
        <div className="fig-plate__tools">
          {plate.training && (
            <Tooltip content="Include the training-set population in the plate">
              <Switch size="sm" checked={training} onChange={onTraining}>Training</Switch>
            </Tooltip>
          )}
          <Button size="sm" icon="download" disabled={off} href={dl("png")} download>PNG</Button>
          <Button size="sm" disabled={off} href={dl("pdf")} download>PDF</Button>
          <Button size="sm" disabled={off} href={dl("svg")} download>SVG</Button>
          <Button asChild size="sm" variant="ghost"><Link to={plate.source.to}>{plate.source.label}</Link></Button>
        </div>
      </header>
      <ServerImage src={preview} alt={plate.title} minHeight={plate.minHeight} className="fig-plate__image"
        onError={() => setFailed(preview)} />
    </section>
  );
}
