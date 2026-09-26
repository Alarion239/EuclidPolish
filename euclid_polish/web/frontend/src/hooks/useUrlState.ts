/* useUrlState — typed search-param state (contract C8).
 *
 *   const [tab, setTab] = useUrlState("tab", "overview");
 *   const [i, setI] = useUrlState("v.main.i", 0);            // number codec
 *   const [tiers, setTiers] = useUrlState<string[]>("tiers", ["lr", "sr"]);
 *
 * The codec is inferred from the default (string / number / boolean as
 * "1"/"0" / string[] as a comma list / anything else as JSON) or given with
 * `{parse, serialize}`. A value equal to the default (same serialization) is
 * REMOVED from the URL, so links stay short. Unparseable params read as the
 * default. History is replaced by default (`{replace: false}` pushes). A
 * write of the value the hook already reads (same serialization — including
 * an explicit default or an unparseable param read as the default) does
 * nothing: in push mode it adds no duplicate entry.
 * Several setters called in one tick are coalesced: each builds on the
 * previous one's search, and the tick records AT MOST ONE new history entry,
 * so one Back undoes the whole tick whatever order the setters ran in. With
 * no push-mode setter the tick edits the current entry in place (keeping its
 * history state). With one, the tick's changes all land in one pushed entry:
 * the first push-mode setter restores the original entry (undoing earlier
 * in-place writes of the tick) and pushes; later setters replace that entry.
 * Other params keep their exact spelling (only this key's segment is
 * rewritten) and the hash is preserved.
 */
import { useCallback, useContext, useLayoutEffect, useMemo, useRef } from "react";
import { UNSAFE_NavigationContext, useLocation, useNavigate } from "react-router-dom";

export type UrlStateOpts<T> = {
  /** raw param → value (return undefined for "invalid → default"). */
  parse?: (raw: string) => T | undefined;
  /** value → raw param (null removes the param). */
  serialize?: (value: T) => string | null;
  /** Replace the history entry (default true). */
  replace?: boolean;
};

type Codec<T> = { parse: (raw: string) => T | undefined; serialize: (v: T) => string | null };

function inferCodec<T>(defaultValue: T): Codec<T> {
  if (typeof defaultValue === "number") {
    return {
      parse: (raw) => { const n = Number(raw); return (raw.trim() !== "" && Number.isFinite(n) ? n : undefined) as T | undefined; },
      serialize: (v) => String(v),
    };
  }
  if (typeof defaultValue === "boolean") {
    return {
      parse: (raw) => (raw === "1" || raw === "true" ? true : raw === "0" || raw === "false" ? false : undefined) as T | undefined,
      serialize: (v) => (v ? "1" : "0"),
    };
  }
  if (typeof defaultValue === "string") {
    return { parse: (raw) => raw as T, serialize: (v) => String(v) };
  }
  if (Array.isArray(defaultValue)) {
    return {
      parse: (raw) => raw.split(",").filter((s) => s !== "") as T,
      serialize: (v) => (v as unknown as string[]).join(","),
    };
  }
  return {
    parse: (raw) => { try { return JSON.parse(raw) as T; } catch { return undefined; } },
    serialize: (v) => JSON.stringify(v),
  };
}

/* The navigation of the current tick, per router (navigator): the search
   written so far (a second setter builds on it before React re-renders),
   the entry's search and history state before the tick (so a push after
   in-place writes can restore that entry first), and whether this tick
   already pushed its one history entry. */
type TickBatch = { search: string; origin: string; originState: unknown; pushed: boolean };
const PENDING = new WeakMap<object, TickBatch>();

/** The decoded key of one raw `k=v` search segment (URLSearchParams rules). */
function segmentKey(segment: string): string {
  const eq = segment.indexOf("=");
  return new URLSearchParams(`${eq < 0 ? segment : segment.slice(0, eq)}=`).keys().next().value ?? "";
}

/** `search` with `key` set to `raw` (null removes it). Only that key's
 *  segments are touched — every other param keeps its exact spelling (no
 *  re-encoding of e.g. `inspect=member:m_1`). The key stays where it first
 *  appeared (duplicates collapse into it), else it is appended. */
function withParam(search: string, key: string, raw: string | null): string {
  const q = search.startsWith("?") ? search.slice(1) : search;
  const own = raw == null ? null : new URLSearchParams([[key, raw]]).toString();
  const out: string[] = [];
  let placed = false;
  for (const segment of q ? q.split("&") : []) {
    if (segment === "") continue;
    if (segmentKey(segment) !== key) { out.push(segment); continue; }
    if (own != null && !placed) { out.push(own); placed = true; }
  }
  if (own != null && !placed) out.push(own);
  return out.length ? `?${out.join("&")}` : "";
}

export function useUrlState<T>(
  key: string,
  defaultValue: T,
  opts: UrlStateOpts<T> = {},
): [T, (next: T | ((prev: T) => T)) => void] {
  const location = useLocation();
  const navigate = useNavigate();
  const { navigator } = useContext(UNSAFE_NavigationContext);

  const inferred = inferCodec(defaultValue);
  const parse = opts.parse ?? inferred.parse;
  const serialize = opts.serialize ?? inferred.serialize;
  const replace = opts.replace ?? true;

  const raw = new URLSearchParams(location.search).get(key);
  const defaultRaw = serialize(defaultValue);

  // Stable value while the raw param and the default's serialization match
  // (callers may pass a fresh default literal on every render).
  const value = useMemo<T>(() => {
    if (raw == null) return defaultValue;
    const parsed = parse(raw);
    return parsed === undefined ? defaultValue : parsed;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [raw, defaultRaw]);

  // Latest location/codec for the (stable) setter, updated after commit.
  const live = useRef({ location, navigate, replace, defaultRaw, parse, serialize, defaultValue });
  useLayoutEffect(() => {
    live.current = { location, navigate, replace, defaultRaw, parse, serialize, defaultValue };
  });

  const setValue = useCallback((next: T | ((prev: T) => T)) => {
    const {
      location: loc, navigate: nav, replace: rep, defaultRaw: dRaw,
      parse: p, serialize: s, defaultValue: d,
    } = live.current;
    let batch = PENDING.get(navigator);
    const base = batch?.search ?? loc.search;
    const currentRaw = new URLSearchParams(base).get(key);
    const prev = currentRaw == null ? d : (p(currentRaw) ?? d);
    const resolved = typeof next === "function" ? (next as (v: T) => T)(prev) : next;
    const out = s(resolved);
    // Writing the value the hook already reads (same serialization: also an
    // explicit default, an unparseable param read as the default, or a
    // non-canonical spelling) is a no-op: nothing to write, and in push mode
    // no duplicate history entry.
    if (out === s(prev)) return;
    const nextRaw = out == null || out === dRaw ? null : out;
    const nextSearch = withParam(base, key, nextRaw);
    if (nextSearch === base) return;
    if (!batch) {
      const fresh: TickBatch = { search: base, origin: base, originState: loc.state, pushed: false };
      batch = fresh;
      PENDING.set(navigator, fresh);
      setTimeout(() => { if (PENDING.get(navigator) === fresh) PENDING.delete(navigator); }, 0);
    }
    const to = (search: string) => ({ pathname: loc.pathname, search, hash: loc.hash });
    if (!rep && !batch.pushed) {
      // The tick's one new entry. Earlier replace-mode writes of this tick
      // edited the original entry in place: put it back first, so that one
      // Back undoes the whole tick whatever order the setters ran in.
      if (batch.search !== batch.origin) nav(to(batch.origin), { replace: true, state: batch.originState });
      nav(to(nextSearch));
      batch.pushed = true;
    } else {
      // In place: the original entry (keeping its history state) or, after
      // this tick's push, the tick's own entry.
      nav(to(nextSearch), { replace: true, state: batch.pushed ? null : batch.originState });
    }
    batch.search = nextSearch;
  }, [key, navigator]);

  return [value, setValue];
}
