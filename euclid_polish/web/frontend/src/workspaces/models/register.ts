/* Registers the Models workspace's inspector kind. The component is a lazy
   chunk, so importing this module is cheap; the workspace imports it (and
   the shell may, so `member:member_257` opens from the palette anywhere):
   `member:<name>` — one universal model (member_257, 257 or 257·psnr). */
import { lazy } from "react";
import { registerInspector } from "../../app/inspector";
import { memberNumber } from "./model";

const MemberInspector = lazy(() => import("./MemberInspector"));

export const unregisterMemberInspector = registerInspector("member", MemberInspector, {
  title: (id) => `Model #${memberNumber(id) ?? id}`,
});
