import { NavLink, Route, Routes, Navigate } from "react-router-dom";
import { NAV } from "./nav";
import { useTheme, type Theme } from "./theme";
import EnsemblePage from "./pages/Ensemble";
import Placeholder from "./pages/Placeholder";
import "./app.css";

function ThemeToggle({ theme, onToggle }: { theme: Theme; onToggle: () => void }) {
  const next = theme === "light" ? "dark" : "light";
  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={onToggle}
      title={`Switch to ${next} theme`}
      aria-label={`Switch to ${next} theme`}
    >
      <span className="theme-toggle__track" data-theme={theme}>
        <span className="theme-toggle__thumb">{theme === "light" ? "☀" : "☾"}</span>
      </span>
      <span className="theme-toggle__label">{theme}</span>
    </button>
  );
}

function Rail({ theme, onToggle }: { theme: Theme; onToggle: () => void }) {
  return (
    <nav className="rail">
      <div className="rail__brand">
        <span className="rail__logo" aria-hidden>◎</span>
        <div>
          <div className="rail__title">EUCLID<span>POLISH</span></div>
          <div className="rail__tag">super-resolution console</div>
        </div>
      </div>
      <div className="rail__scroll">
        {NAV.map((sec) => (
          <div className="rail__sec" key={sec.title}>
            <div className="eyebrow rail__sechead">{sec.title}</div>
            {sec.items.map((it) =>
              it.to ? (
                <NavLink key={it.label} to={it.to}
                  className={({ isActive }) => `rail__item ${isActive ? "is-active" : ""}`}>
                  <span>{it.label}</span>
                  {it.done && <span className="rail__badge">new</span>}
                </NavLink>
              ) : (
                <a key={it.label} href={it.legacy} className="rail__item rail__item--legacy">
                  <span>{it.label}</span>
                  <span className="rail__ext" title="legacy page">↗</span>
                </a>
              ),
            )}
          </div>
        ))}
      </div>
      <div className="rail__foot">
        <a href="/" className="rail__foot-link">← classic UI</a>
        <ThemeToggle theme={theme} onToggle={onToggle} />
      </div>
    </nav>
  );
}

export default function App() {
  const { theme, toggle } = useTheme();
  return (
    <div className="shell">
      <Rail theme={theme} onToggle={toggle} />
      <main className="stage">
        <Routes>
          <Route path="/" element={<Navigate to="/ensemble" replace />} />
          <Route path="/ensemble" element={<EnsemblePage theme={theme} />} />
          <Route path="*" element={<Placeholder />} />
        </Routes>
      </main>
    </div>
  );
}
