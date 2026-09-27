/* "Cache a 25.6″ real tile": RA/Dec (degrees or sexagesimal) → POST
 * /api/real/tiles after a Q1 coverage check (./actions.cacheTile). */
import { useState, type ReactElement } from "react";
import { formatDec, formatRA, parseSkyCoord } from "../../../format";
import { Button, Checkbox, Field, Input, Popover } from "../../../ui";
import { cacheTile } from "./actions";

export function CacheTilePopover({ trigger, open: openProp, onOpenChange }: {
  trigger?: ReactElement; open?: boolean; onOpenChange?: (open: boolean) => void;
}) {
  const [inner, setInner] = useState(false);
  const open = openProp ?? inner;
  const setOpen = (v: boolean) => { setInner(v); onOpenChange?.(v); };
  const [text, setText] = useState("");
  const [run, setRun] = useState(true);
  const [busy, setBusy] = useState(false);
  const coord = parseSkyCoord(text);
  const bad = text.trim() !== "" && (!coord || coord.ra < 0 || coord.ra >= 360 || coord.dec < -90 || coord.dec > 90);
  const submit = async () => {
    if (!coord || bad) return;
    setBusy(true);
    try {
      const r = await cacheTile(coord.ra, coord.dec, { run });
      if (r?.ok) setOpen(false);
    } finally { setBusy(false); }
  };
  return (
    <Popover open={open} onOpenChange={setOpen} label="Cache a 25.6″ real tile" width={340} align="end"
      trigger={trigger ?? <Button size="sm" icon="plus">Cache tile…</Button>}>
      <div className="res-form">
        <Field label="Position" description={coord && !bad ? `${formatRA(coord.ra)} ${formatDec(coord.dec)}` : "RA Dec in degrees or hh:mm:ss ±dd:mm:ss"}
          error={bad ? "Not a sky position (RA 0–360°, Dec ±90°)" : undefined}>
          <Input value={text} onChange={setText} onEnter={submit} placeholder="273.2309 68.3637" autoFocus />
        </Field>
        <Checkbox checked={run} onChange={setRun}>Then run production + mean</Checkbox>
        <div className="res-form__foot">
          <Button size="sm" variant="primary" loading={busy} disabled={!coord || bad} onClick={submit}>Cache tile</Button>
        </div>
      </div>
    </Popover>
  );
}
