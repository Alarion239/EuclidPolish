/* Fallback for routes not yet ported into the SPA. Keeps the console navigable
   while the redesign rolls out tab by tab. */
import { useLocation } from "react-router-dom";
import { Card, CardBody, Empty } from "../ui";

export default function Placeholder() {
  const loc = useLocation();
  return (
    <div className="page">
      <header className="page__head">
        <div>
          <div className="eyebrow">console</div>
          <h1 className="page__title">Not ported yet</h1>
          <div className="page__sub">
            <code className="mono">{loc.pathname}</code> hasn’t been rebuilt in the new
            console yet.
          </div>
        </div>
      </header>
      <Card>
        <CardBody>
          <Empty>
            This tab still lives in the classic UI. Use the sidebar’s ↗ links, or go
            back to the <a href="/">classic UI</a>.
          </Empty>
        </CardBody>
      </Card>
    </div>
  );
}
