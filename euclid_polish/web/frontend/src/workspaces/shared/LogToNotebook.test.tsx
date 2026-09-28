/* "Log to notebook" (workspaces/shared/LogToNotebook.tsx): every such button
 * lands on Notebook › Log with the entry prefilled — nothing is appended
 * from the page itself. */
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { UiProvider } from "../../ui";
import { LogToNotebookButton, useLogToNotebook } from "./LogToNotebook";
import { notebookEntryUrl } from "./noteText";

function LocationProbe() {
  const loc = useLocation();
  return <output data-testid="loc">{loc.pathname + loc.search}</output>;
}

const show = (el: JSX.Element) => render(
  <MemoryRouter initialEntries={["/models/starfull/leaderboard"]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
    <UiProvider>{el}</UiProvider><LocationProbe />
  </MemoryRouter>,
);

describe("notebookEntryUrl", () => {
  it("lands on Notebook › Log with the entry and its page", () => {
    expect(notebookEntryUrl("## Gate fit\n- +0.14 dB", "Models › Combiner"))
      .toBe("/notebook/log?entry=%23%23+Gate+fit%0A-+%2B0.14+dB&from=Models+%E2%80%BA+Combiner");
    expect(notebookEntryUrl("x")).toBe("/notebook/log?entry=x");
  });
});

describe("LogToNotebookButton", () => {
  it("builds the note on click and opens Notebook › Log prefilled", () => {
    const note = vi.fn(() => "## Knee leaderboard\n- production 44.1 dB");
    show(<LogToNotebookButton note={note} from="Models › Leaderboard" />);
    expect(note).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Log to notebook" }));
    expect(note).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("loc").textContent)
      .toBe(notebookEntryUrl("## Knee leaderboard\n- production 44.1 dB", "Models › Leaderboard"));
  });

  it("does nothing while disabled or for an empty note", () => {
    show(<>
      <LogToNotebookButton note={() => "x"} from="Sky › Compare" disabled />
      <LogToNotebookButton note={() => "   "} from="Sky › Compare" label="Log empty" />
    </>);
    fireEvent.click(screen.getByRole("button", { name: "Log to notebook" }));
    fireEvent.click(screen.getByRole("button", { name: "Log empty" }));
    expect(screen.getByTestId("loc").textContent).toBe("/models/starfull/leaderboard");
  });
});

describe("useLogToNotebook", () => {
  it("returns a function that opens the notebook with that entry", () => {
    function Menu() {
      const log = useLogToNotebook("Home");
      return <button type="button" onClick={() => log("## Catch-up")}>Log it</button>;
    }
    show(<Menu />);
    fireEvent.click(screen.getByRole("button", { name: "Log it" }));
    expect(screen.getByTestId("loc").textContent).toBe("/notebook/log?entry=%23%23+Catch-up&from=Home");
  });
});
