import React from "react";
import { createRoot } from "react-dom/client";
import { QueryClientProvider } from "@tanstack/react-query";
import App from "./app/App";
import { queryClient } from "./api/query";
import { bindPrefsToDocument } from "./state/prefs";
import "./theme/index.css";

// Resolved theme / accent / density onto <html> (the pre-paint script in
// index.html already applied the saved values before first paint).
bindPrefsToDocument();

// The data router lives in app/App.tsx; UiProvider (tooltips, toasts, the
// confirm() host) is mounted once by the shell INSIDE the router's root
// layout route, so their content may use <Link> (FOUNDATION §9.1).
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
);
