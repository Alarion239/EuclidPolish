/* Fallback for optional/experimental routes without a dedicated React page. */
import { useLocation } from "react-router-dom";
import { Card, CardBody, Empty } from "../ui";

export default function Placeholder() {
  const loc = useLocation();
  return (
    <div className="page">
      <header className="page__head">
        <div>
          <div className="eyebrow">console</div>
          <h1 className="page__title">Unavailable in this build</h1>
          <div className="page__sub">
            <code className="mono">{loc.pathname}</code> hasn’t been rebuilt in the new
            console yet.
          </div>
        </div>
      </header>
      <Card>
        <CardBody>
          <Empty>
            This route is reserved for an optional pipeline lane and is not enabled
            in the current React console build.
          </Empty>
        </CardBody>
      </Card>
    </div>
  );
}
