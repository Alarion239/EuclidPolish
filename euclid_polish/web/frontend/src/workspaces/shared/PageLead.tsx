/* A page's lead line (Ops, Settings, Home): the explanatory text with the
 * page's own actions at its right. It replaces the per-tab visible title
 * and eyebrow: the breadcrumb and the active tab already name the page, and
 * <Workspace> renders the page's (visually hidden) h1. */
import type { ReactNode } from "react";
import "./shared.css";

export function PageLead({ children, right }: { children?: ReactNode; right?: ReactNode }) {
  return (
    <div className="page-lead">
      {children != null && <p className="page-lead__text">{children}</p>}
      {right != null && <div className="page-lead__right">{right}</div>}
    </div>
  );
}
