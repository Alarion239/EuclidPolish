/* Navigation + export row (as the old engine): ◀ index ▶, ▶▶ run-through
 * with its speed, the key hint, ⬇ PNG, ⬇ Figure, Save crop to results (S),
 * the save status and ⏺ video. */
import { useEffect, useState } from "react";
import { useController, useViewer } from "./hooks";

export function Nav({ onPng, onFigure, onRecord }: { onPng: () => void; onFigure: () => void; onRecord: () => void }) {
  const ctrl = useController();
  const meta = useViewer((s) => s.meta);
  const index = useViewer((s) => s.index);
  const playing = useViewer((s) => s.playing);
  const playMs = useViewer((s) => s.playMs);
  const save = useViewer((s) => s.save);
  const saveInFlight = useViewer((s) => s.saveInFlight);
  const recording = useViewer((s) => s.recording);
  useViewer((s) => s.frozen);
  useViewer((s) => s.status);
  const [draft, setDraft] = useState<string | null>(null);
  useEffect(() => setDraft(null), [index]);
  if (!meta) return null;
  const count = meta.count;
  const obj = meta.objects?.[index];
  const reason = ctrl.saveBlockReason();
  const commit = () => {
    const v = parseInt(draft ?? "", 10);
    setDraft(null);
    if (Number.isFinite(v)) ctrl.go(v, false);   // explicit jump → clamp, don't wrap
  };
  return (
    <div className="cv-nav">
      <button type="button" className="cv-navbtn" title="Previous (←)" aria-label="Previous" onClick={() => ctrl.go(index - 1)}>◀</button>
      <span className="cv-idx">
        <input className="cv-idx-input" type="text" inputMode="numeric" aria-label="Index"
          title="type an index and press Enter to jump" value={draft ?? String(index)}
          onChange={(e) => setDraft(e.target.value)} onBlur={commit}
          onKeyDown={(e) => { if (e.key === "Enter") { commit(); (e.target as HTMLInputElement).blur(); } }} />
        <span className="cv-idx-total"> / {Math.max(0, count - 1)}</span>
      </span>
      <button type="button" className="cv-navbtn" title="Next (→)" aria-label="Next" onClick={() => ctrl.go(index + 1)}>▶</button>
      <button type="button" className={`cv-navbtn cv-play${playing ? " active" : ""}`} title="Run through (Space)"
        aria-pressed={playing} onClick={() => ctrl.togglePlay()}>▶▶</button>
      <label className="cv-slider cv-speed" title="auto-run cadence — seconds per slide">
        <span>speed</span>
        <input type="range" className="cv-range" min={0} max={1000}
          value={Math.round((1000 * (Math.log(playMs / 1000) - Math.log(0.3))) / (Math.log(3) - Math.log(0.3)))}
          onChange={(e) => ctrl.setPlaySpeed(1000 * Math.exp(Math.log(0.3) + (Number(e.target.value) / 1000) * (Math.log(3) - Math.log(0.3))))}
          aria-label="Run-through speed" />
        <span className="cv-val">{(playMs / 1000).toFixed(1)} s</span>
      </label>
      {obj?.label && <span className="cv-objlabel" title={obj.id}>{obj.label}</span>}
      <span className="cv-kbd">← →  ·  Space to run  ·  S save  ·  +/− zoom</span>
      <button type="button" className="cv-navbtn" title="Save the current view (all selected tiers, side by side) as a PNG" onClick={onPng}>⬇ PNG</button>
      <button type="button" className="cv-navbtn cv-figure"
        title="Export a high-resolution publication plate. A selected magnification region exports as matched crops; without one, full images are exported."
        onClick={onFigure}>⬇ Figure</button>
      <button type="button" className="cv-navbtn cv-save-result" disabled={!!reason}
        title={reason || "Save the frozen matched raw cubes and manifest to the results area"}
        onClick={() => { void ctrl.saveCropToResults(); }}>{saveInFlight ? "Saving crop…" : "Save crop to results"}</button>
      <span className="cv-save-status" role="status" aria-live="polite" data-tone={save.tone}>{save.text}</span>
      <button type="button" className={`cv-navbtn cv-rec${recording ? " cv-recording" : ""}`}
        title="Record the current view (all selected tiers, side by side) — click to start, click again to stop and download a .webm clip"
        onClick={onRecord}>{recording ? "⏹ stop" : "⏺ video"}</button>
    </div>
  );
}
