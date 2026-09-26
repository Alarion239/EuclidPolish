/* App-level hosts for the kit: the shared tooltip provider, the toast
   stack and the confirm() dialog host. Mount once around the app, INSIDE the
   router and the QueryClientProvider (main.tsx; with the WP-F3 data router:
   inside the root layout route), so content rendered by these hosts (toast
   actions, confirm messages, tooltips) can use <Link> and context hooks. */
import type { ReactNode } from "react";
import { ConfirmHost } from "./confirm";
import { TooltipProvider } from "./overlays";
import { Toaster } from "./toast";

export function UiProvider({ children }: { children: ReactNode }) {
  return (
    <TooltipProvider>
      {children}
      <Toaster />
      <ConfirmHost />
    </TooltipProvider>
  );
}
