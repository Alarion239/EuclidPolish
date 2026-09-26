import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Unit tests (vitest + happy-dom + testing-library). Tests live next to the
// module they cover (`src/**/*.test.ts[x]`); `test/` holds suites ported from
// the old node:test runner. See src/FOUNDATION.md → "Testing".
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "happy-dom",
    include: ["src/**/*.test.{ts,tsx}", "test/**/*.test.{ts,tsx}"],
    setupFiles: ["./test/setup.ts"],
    restoreMocks: true,
    unstubGlobals: true,
    clearMocks: true,
    // Stylesheets are not processed in tests, except `?raw` text imports
    // (used by the token-contract tests to read the CSS as a string).
    css: { include: [/\?raw$/] },
    // Node ≥ 25 ships its own (file-backed, here unconfigured) global
    // localStorage that shadows happy-dom's; turn it off so tests get the DOM
    // Storage. Harmless on Node 22, where the flag is off by default.
    poolOptions: {
      forks: { execArgv: ["--no-experimental-webstorage"] },
      threads: { execArgv: ["--no-experimental-webstorage"] },
    },
  },
});
