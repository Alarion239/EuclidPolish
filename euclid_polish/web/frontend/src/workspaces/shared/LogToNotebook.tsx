/* "Log to notebook", shared by every workspace (Models › Leaderboard and
 * Combiner, Sky › Compare, Home's "no entry since" alert): the page builds a
 * markdown entry from its facts and the button lands on Notebook › Log with
 * that entry prefilled (`?entry=&from=`, `noteText.ts notebookEntryUrl`).
 * Nothing is appended from the page: the notebook's own Append does it,
 * after the entry was read and edited there.
 *
 *   <LogToNotebookButton note={() => markdown} from="Models › Leaderboard" />
 *   const log = useLogToNotebook("Home"); … log(markdown)   // a menu item, a palette action
 */
import { useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Button, type ButtonSize } from "../../ui";
import { notebookEntryUrl } from "./noteText";

/** `(entry) => void`: open Notebook › Log with `entry` prefilled, naming the
 *  page it came from. A blank entry does nothing. */
export function useLogToNotebook(from: string): (entry: string) => void {
  const navigate = useNavigate();
  return useCallback((entry: string) => {
    if (entry.trim()) navigate(notebookEntryUrl(entry, from));
  }, [navigate, from]);
}

export function LogToNotebookButton({ note, from, disabled, size = "sm", label = "Log to notebook", title }: {
  /** Builds the entry on click (the page may still be updating). */
  note: () => string;
  /** The page label the notebook names as the entry's source ("Sky › Compare"). */
  from: string;
  disabled?: boolean; size?: ButtonSize; label?: string; title?: string;
}) {
  const log = useLogToNotebook(from);
  return (
    <Button size={size} icon="pin" disabled={disabled} title={title ?? "Open Notebook › Log with this entry prefilled; you edit it there before it is appended"}
      onClick={() => log(note())}>{label}</Button>
  );
}
