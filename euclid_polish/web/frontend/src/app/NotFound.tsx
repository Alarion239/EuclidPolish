/* Unknown URL inside the SPA (a mistyped path, a removed page, a tab the
 * manifest does not list). Suggests the workspace the path starts with. */
import { Link, useLocation } from "react-router-dom";
import { Button, EmptyState, Page } from "../ui";
import { MANIFEST } from "./manifest";
import { landingPath } from "./nav";

export function NotFound() {
  const { pathname } = useLocation();
  const first = pathname.split("/").filter(Boolean)[0] ?? "";
  const near = MANIFEST.workspaces.find((w) => w.path.split("/").filter(Boolean)[0] === first && w.id !== "home");
  return (
    <Page>
      <EmptyState icon="search" title="No page here"
        action={(
          <div className="row" style={{ gap: "var(--s2)", justifyContent: "center" }}>
            {near && <Button asChild variant="primary" size="sm"><Link to={landingPath(near.id)}>Open {near.label}</Link></Button>}
            <Button asChild size="sm"><Link to="/">Home</Link></Button>
          </div>
        )}>
        <code className="mono">{pathname}</code> is not a page of this console. Press
        {" "}<kbd className="ui-kbd">⌘K</kbd> / <kbd className="ui-kbd">Ctrl K</kbd> to search every page.
      </EmptyState>
    </Page>
  );
}
