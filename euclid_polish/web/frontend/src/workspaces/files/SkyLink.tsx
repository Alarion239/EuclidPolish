/* "Show on sky": a router link to the Sky atlas centred on an image's WCS
   (`/sky/atlas?ra&dec&fov`), styled as a small button with its icon. */
import { Link } from "react-router-dom";
import { formatRaDec } from "../../format";
import { Button, Icon } from "../../ui";
import type { WcsSummary } from "./api";
import { skyHref } from "./model";

export function SkyLink({ wcs, label = "Show on sky", className }: {
  wcs: WcsSummary; label?: string; className?: string;
}) {
  return (
    <Button asChild size="sm" className={className}>
      <Link to={skyHref(wcs)} title={`Open the Sky atlas at ${formatRaDec(wcs.ra, wcs.dec)}`}>
        <Icon name="globe" />
        <span className="ui-btn__label">{label}</span>
      </Link>
    </Button>
  );
}
