/* Toasts (sonner). `toast("Saved")`, `toast.success(...)`, `toast.error(...)`,
   `toast.warning(...)`, `toast.info(...)`, `toast.promise(p, {...})`,
   `toast.dismiss(id)`. <Toaster/> is mounted once by UiProvider and follows
   the app theme; its colours come from the tokens (ui.css). */
import { Toaster as Sonner, toast } from "sonner";
import { useResolvedTheme } from "../state/prefs";

export { toast };

export function Toaster() {
  const theme = useResolvedTheme();
  return (
    <Sonner theme={theme} position="bottom-right" closeButton visibleToasts={5}
      className="ui-toaster" toastOptions={{ className: "ui-toast", duration: 5000 }} />
  );
}
