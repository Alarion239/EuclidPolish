/* Global vitest setup: unmount rendered trees and reset shared foundation
   state between tests so suites cannot leak into each other. */
import { cleanup, configure } from "@testing-library/react";
import { afterEach } from "vitest";

// findBy*/waitFor give up after 3 s, not testing-library's 1 s: on a loaded
// machine a wait that takes 30–80 ms idle took up to 1.8 s (a role query
// re-runs over the whole page on every DOM change). A wrong state still
// fails, only later.
configure({ asyncUtilTimeout: 3000 });

afterEach(() => {
  cleanup();
  try { window.localStorage.clear(); } catch { /* storage unavailable */ }
});
