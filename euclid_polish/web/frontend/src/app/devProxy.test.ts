import { describe, expect, it } from "vitest";
import { devRoute } from "./devProxy";

describe("dev proxy routing", () => {
  it("serves page paths and legacy page URLs from the SPA", () => {
    for (const u of ["/", "/sky", "/sky/atlas?ra=1", "/ensemble/starless/train", "/config", "/noise?x=1", "/inspect"])
      expect(devRoute(u), u).toBe("spa");
  });

  it("leaves Vite's own module URLs to Vite", () => {
    for (const u of ["/@vite/client", "/@react-refresh", "/src/main.tsx", "/node_modules/.vite/deps/react.js", "/@fs/x/y.ts", "/@id/virtual", "/index.html"])
      expect(devRoute(u), u).toBe("vite");
  });

  it("proxies every non-page path to Flask", () => {
    for (const u of [
      "/api/jobs", "/ensemble/status.json?mode=starfull", "/viewer/meta/sky", "/static/dist/index.html",
      "/auth/status", "/euclid-auth/login", "/inspect/preview.png?fits=a", "/cutout-image/1", "/pix/x.png",
      "/sky/unknown", "/app/sky",
    ]) expect(devRoute(u), u).toBe("flask");
  });
});
