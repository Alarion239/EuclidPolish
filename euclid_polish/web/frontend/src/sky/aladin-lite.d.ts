/* aladin-lite ships no types; `src/sky/types.ts` describes the slice we use.
 * Import it ONLY through `loadAladin()` (src/sky/engine.ts): a dynamic import
 * keeps the 2.4 MB bundle (and its WASM compile) out of the main chunk. */
declare module "aladin-lite" {
  const A: unknown;
  export default A;
}
