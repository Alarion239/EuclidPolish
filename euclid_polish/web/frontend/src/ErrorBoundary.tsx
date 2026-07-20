/* Per-route error boundary: a runtime error in one page shows a contained
   message instead of blanking the whole console. Reset when the route changes
   so navigating away recovers. */
import { Component, type ReactNode } from "react";

type Props = { routeKey: string; children: ReactNode };
type State = { error: Error | null };

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State { return { error }; }

  componentDidUpdate(prev: Props) {
    if (prev.routeKey !== this.props.routeKey && this.state.error) this.setState({ error: null });
  }

  render() {
    if (this.state.error) {
      return (
        <div className="page">
          <header className="page__head">
            <div>
              <div className="eyebrow">console · error</div>
              <h1 className="page__title">This page hit a snag</h1>
              <div className="page__sub">
                It rendered an unexpected value. Try another tab, or reload this page.
              </div>
            </div>
          </header>
          <section className="ui-card"><div className="ui-card__body">
            <pre className="ui-logtail">{String(this.state.error?.message || this.state.error)}</pre>
          </div></section>
        </div>
      );
    }
    return this.props.children;
  }
}
