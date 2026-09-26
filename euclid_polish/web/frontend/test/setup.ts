/* Global vitest setup: unmount rendered trees and reset shared foundation
   state between tests so suites cannot leak into each other. */
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
  try { window.localStorage.clear(); } catch { /* storage unavailable */ }
});
