/* Registers the Models workspace's inspector kinds. The components are lazy
   chunks, so importing this module is cheap; the workspace imports it (and
   the shell may, so `member:member_196` opens from the palette anywhere).
   - `member:<name>` — one member (member_196, 196 or 196·psnr);
   - `combiner:<regime>/<variant dir>` — one combiner variant. */
import { lazy } from "react";
import { registerInspector } from "../../app/inspector";
import { memberNumber } from "./model";

const MemberInspector = lazy(() => import("./MemberInspector"));
const CombinerInspector = lazy(() => import("./CombinerInspector"));

export const unregisterMemberInspector = registerInspector("member", MemberInspector, {
  title: (id) => `Member #${memberNumber(id) ?? id}`,
});

export const unregisterCombinerInspector = registerInspector("combiner", CombinerInspector, {
  title: (id) => `Combiner ${id.split("/").slice(1).join("/").replace(/^spatial_gate_/, "") || id}`,
});
