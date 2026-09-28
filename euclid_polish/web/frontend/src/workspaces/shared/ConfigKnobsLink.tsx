/* "N knobs changed · Edit": the link back to System › Config on a tab that
 * judges a config group's effect (Synthetic › Records, Synthetic › PSF,
 * Models › Train). N counts that group's knobs that differ from their
 * defaults (GET /api/config, read-only); nothing shows while the config
 * loads, when it fails, or when every knob is at its default.
 *
 *   <ConfigKnobsLink groups={["scenes", "lenses"]} />
 */
import { Link } from "react-router-dom";
import { useResource } from "../../api/query";
import { pagePath } from "../../app/nav";
import { changedKnobs, type ConfigValues } from "../system/configModel";
import type { GroupId } from "../system/configFields";

type ConfigResp = { config?: ConfigValues; defaults?: ConfigValues; types?: Record<string, string> };

/** The Config link for these groups: `?group=<first>&changed=1`. */
export function configGroupUrl(groups: readonly GroupId[]): string {
  const q = new URLSearchParams({ ...(groups.length === 1 ? { group: groups[0] } : {}), changed: "1" });
  return `${pagePath("system", { tab: "config" })}?${q.toString()}`;
}

export function ConfigKnobsLink({ groups, className }: { groups: readonly GroupId[]; className?: string }) {
  const cfg = useResource<ConfigResp>("/api/config", [], { ttl: 60_000 });
  const d = cfg.data;
  if (!d?.config || !d.defaults) return null;
  const n = changedKnobs(d.config, d.defaults, groups, d.types);
  if (!n) return null;
  return (
    <Link className={className} to={configGroupUrl(groups)}
      title="The knobs of this tab's config group that differ from their defaults, in System › Config">
      {n} knob{n === 1 ? "" : "s"} changed · Edit
    </Link>
  );
}
