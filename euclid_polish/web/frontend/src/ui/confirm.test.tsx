import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { ConfirmHost, confirm, resetConfirm } from "./confirm";

afterEach(() => act(() => resetConfirm()));

/** Opens a confirmation; returns its pending promise wrapped (an async
 *  function returning the promise itself would wait for the answer). */
async function openConfirm(...args: Parameters<typeof confirm>): Promise<{ answer: Promise<boolean> }> {
  let answer!: Promise<boolean>;
  await act(async () => { answer = confirm(...args); });
  return { answer };
}

describe("confirm()", () => {
  it("auto-mounts a host and resolves true on Confirm", async () => {
    const { answer: p } = await openConfirm({ title: "Archive 3 members?", message: "They move to tracking.", confirmLabel: "Archive" });
    const dialog = await screen.findByRole("alertdialog", { name: "Archive 3 members?" });
    expect(dialog.textContent).toContain("They move to tracking.");
    // the message is the dialog's accessible description (announced with the title)
    const described = dialog.getAttribute("aria-describedby");
    expect(described).toBeTruthy();
    expect(document.getElementById(described!)?.textContent).toBe("They move to tracking.");
    fireEvent.click(screen.getByRole("button", { name: "Archive" }));
    await expect(p).resolves.toBe(true);
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull());
  });

  it("has no dangling aria-describedby without a message", async () => {
    await openConfirm("Push to origin?");
    const dialog = await screen.findByRole("alertdialog", { name: "Push to origin?" });
    const described = dialog.getAttribute("aria-describedby");
    if (described) expect(document.getElementById(described)).toBeTruthy();
  });

  it("resolves false on Cancel and on Escape", async () => {
    const { answer: p1 } = await openConfirm("Delete?");
    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    await expect(p1).resolves.toBe(false);

    const { answer: p2 } = await openConfirm({ title: "Push to origin?" });
    const dialog = await screen.findByRole("alertdialog", { name: "Push to origin?" });
    fireEvent.keyDown(dialog, { key: "Escape" });
    await expect(p2).resolves.toBe(false);
  });

  it("focuses Cancel first for a danger confirmation", async () => {
    await openConfirm({ title: "Delete outputs?", tone: "danger", confirmLabel: "Delete" });
    await screen.findByRole("alertdialog");
    await waitFor(() => expect(document.activeElement?.textContent).toBe("Cancel"));
    expect(screen.getByRole("button", { name: "Delete" }).className).toContain("ui-btn--danger");
  });

  it("requires the exact text before confirming when requireText is set", async () => {
    const { answer: p } = await openConfirm({ title: "Sync with --delete-after?", requireText: "sync" });
    const ok = await screen.findByRole("button", { name: "Confirm" });
    expect((ok as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByRole("textbox", { name: "Confirmation text" }), { target: { value: "sync" } });
    expect((ok as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(ok);
    await expect(p).resolves.toBe(true);
  });

  it("queues concurrent requests and shows them one at a time", async () => {
    const { answer: a } = await openConfirm("First?");
    const { answer: b } = await openConfirm("Second?");
    expect(screen.getAllByRole("alertdialog")).toHaveLength(1);
    await screen.findByRole("alertdialog", { name: "First?" });
    fireEvent.click(screen.getByRole("button", { name: "Confirm" }));
    await expect(a).resolves.toBe(true);
    await screen.findByRole("alertdialog", { name: "Second?" });
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await expect(b).resolves.toBe(false);
  });

  it("uses an explicitly mounted host (no duplicate dialog)", async () => {
    render(<ConfirmHost />);
    const { answer: p } = await openConfirm("Only once?");
    await screen.findByRole("alertdialog", { name: "Only once?" });
    expect(screen.getAllByRole("alertdialog")).toHaveLength(1);
    expect(document.querySelector("[data-ui-confirm-host]")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Confirm" }));
    await expect(p).resolves.toBe(true);
  });
});
