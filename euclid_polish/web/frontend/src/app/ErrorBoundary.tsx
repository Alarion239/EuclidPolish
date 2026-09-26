/* Error boundaries.
 *
 *   <ErrorBoundary resetKey={pathname} label="Ensemble › Members">…</ErrorBoundary>
 *
 * A runtime error in one tab shows a contained card (message, Retry, Copy
 * details, Reload) instead of blanking the console; the shell, rail and
 * other workspaces keep working. The boundary resets when `resetKey` changes
 * (the shell passes the pathname, so navigating away recovers). `RouteError`
 * is the data router's `errorElement` for errors outside any tab boundary.
 */
import { Component, type ErrorInfo, type ReactNode } from "react";
import { isRouteErrorResponse, useRouteError } from "react-router-dom";
import { Button, Callout, CopyButton } from "../ui";

function describe(error: unknown): { message: string; stack: string } {
  if (isRouteErrorResponse(error)) {
    return { message: `${error.status} ${error.statusText}`.trim(), stack: String(error.data ?? "") };
  }
  if (error instanceof Error) return { message: error.message || error.name, stack: error.stack ?? "" };
  return { message: String(error), stack: "" };
}

/** Text for "Copy details": message, where, when, stacks. */
export function errorDetails(error: unknown, componentStack?: string | null): string {
  const { message, stack } = describe(error);
  const where = typeof window !== "undefined" ? window.location.href : "";
  return [
    `Error: ${message}`,
    `URL: ${where}`,
    `Time: ${new Date().toISOString()}`,
    stack && `\nStack:\n${stack}`,
    componentStack && `\nComponent stack:${componentStack}`,
  ].filter(Boolean).join("\n");
}

export function ErrorView(
  { error, componentStack, onRetry, label }: {
    error: unknown; componentStack?: string | null; onRetry?: () => void; label?: string;
  },
) {
  const { message } = describe(error);
  // A failed lazy chunk (the bundle was rebuilt under a running page) only
  // recovers with a reload: React caches the rejected import.
  const chunk = /dynamically imported module|Loading chunk|Importing a module script failed/i.test(message);
  return (
    <div className="page shell-error">
      <Callout tone="bad" title={label ? `${label} hit an error` : "This page hit an error"}
        action={(
          <div className="row" style={{ gap: "var(--s2)" }}>
            {onRetry && !chunk && <Button size="sm" onClick={onRetry}>Retry</Button>}
            <Button size="sm" variant={chunk ? "primary" : "default"} onClick={() => window.location.reload()}>Reload page</Button>
            <CopyButton value={() => errorDetails(error, componentStack)} label="Copy details" showLabel />
          </div>
        )}>
        <p style={{ margin: 0 }}>
          {chunk
            ? "The console was rebuilt while this page was open. Reload to load the new version."
            : "Other workspaces still work. Retry, or copy the details for a bug report."}
        </p>
        <pre className="shell-error__msg">{message}</pre>
      </Callout>
    </div>
  );
}

type Props = { resetKey: string; label?: string; children: ReactNode };
type State = { error: unknown; componentStack: string | null; key: string };

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, componentStack: null, key: this.props.resetKey };

  static getDerivedStateFromError(error: unknown): Partial<State> {
    return { error: error ?? new Error("unknown error") };
  }

  static getDerivedStateFromProps(props: Props, state: State): Partial<State> | null {
    if (props.resetKey !== state.key) return { key: props.resetKey, error: null, componentStack: null };
    return null;
  }

  componentDidCatch(_error: unknown, info: ErrorInfo) {
    this.setState({ componentStack: info.componentStack ?? null });
  }

  retry = () => this.setState({ error: null, componentStack: null });

  render() {
    if (this.state.error != null) {
      return <ErrorView error={this.state.error} componentStack={this.state.componentStack}
        onRetry={this.retry} label={this.props.label} />;
    }
    return this.props.children;
  }
}

/** Data-router `errorElement`: an error outside every tab boundary. */
export function RouteError() {
  const error = useRouteError();
  return <ErrorView error={error} />;
}
