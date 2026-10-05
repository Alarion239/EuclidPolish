/* App-level hosts for the kit: the shared tooltip provider, the toast
   stack and the confirm() dialog host. Mounted once, by the shell
   (app/Shell.tsx), INSIDE the data router's root layout route and inside the
   QueryClientProvider (main.tsx), so content rendered by these hosts (toast
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
