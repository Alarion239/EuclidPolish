/* Small shared pieces of the Models tabs: the error / empty states that show
   the server's own message, facet colours, the colour-by control and the
   gate-share bar. Tab control bars are the kit's Toolbar. */
import { type ReactNode } from "react";
import type { ApiError } from "../../api/client";
import { openInspector } from "../../app/inspector";
import { Button, EmptyState, Skeleton } from "../../ui";

/** Loading → skeleton; error → the server's message with a retry; empty → `empty`. */
export function LoadState(
  { loading, error, onRetry, empty, children, lines = 4 }: {
    loading: boolean; error: ApiError | null; onRetry?: () => void; empty?: ReactNode | false;
    children?: ReactNode; lines?: number;
  },
) {
  if (loading) return <Skeleton lines={lines} />;
  if (error) {
    return (
      <EmptyState icon="warn" title={error.status === 404 ? "Nothing here yet" : "Could not load"}
        action={onRetry && <Button size="sm" icon="reset" onClick={onRetry}>Retry</Button>}>
        <span className="mdl-mono">{error.message}</span>
      </EmptyState>
    );
  }
  if (empty) return <>{empty}</>;
  return <>{children}</>;
}

export const openMember = (name: string) => openInspector({ kind: "member", id: name });
