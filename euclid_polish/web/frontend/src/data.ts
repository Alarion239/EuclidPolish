/** Runtime-safe collection boundary for JSON responses.
 *
 * TypeScript describes the payload we expect, but it cannot guarantee what an
 * older server, a partial artifact, or a hand-written JSON file actually
 * returns.  Pages should normalize collection fields before mapping/filtering
 * them so an unavailable optional feature remains an empty state instead of
 * taking down the route.
 */
export function asArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? value as T[] : [];
}
