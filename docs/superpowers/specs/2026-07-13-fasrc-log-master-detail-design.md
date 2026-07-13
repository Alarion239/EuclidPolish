# FASRC Past-Run Log Master/Detail Design

## Goal

Make raw logs on `/app/fasrc` easy to inspect without scrolling below the past-runs table. Selecting a run replaces the list card with that run's log viewer, and a Back action restores the list.

## Interaction

- The Logs tab initially shows the existing paginated Past runs table.
- Every run row with at least one available log file is interactive by mouse and keyboard.
- Selecting a run replaces the entire Past runs card with a log-detail card in the same location.
- The detail view opens `.out` by default. If `.out` is unavailable, it falls back to `.err`.
- `.out` and `.err` controls switch the displayed file. A control is disabled when its file is unavailable.
- `Back to past runs` restores the same runs-list page without refetching solely because the view changed.
- The log content scrolls inside a viewport sized for the page, so opening or reading a log does not require scrolling past the runs table.

## Data Flow

The existing backend APIs remain unchanged:

1. `/api/fasrc/runs` supplies each run's metadata and `.out` / `.err` paths.
2. Selection stores the chosen run and preferred file type in React state.
3. `/api/fasrc/runs/log` supplies one paginated log window.
4. Changing file type resets the log page to the newest window.
5. Returning to the list clears only the selection; the runs page remains unchanged.

## States and Errors

- While a log window is loading, the detail card shows an explicit loading state.
- Empty files show `(empty)`.
- Fetch failures appear in the detail card and do not discard the selected run.
- Rows with no available `.out` or `.err` remain visible but are not presented as actionable.
- Existing older/newer/newest pagination is retained for both the run list and log contents.

## Visual and Accessibility Treatment

The feature reuses the current FASRC card, button, badge, mono-log, focus, and theme tokens. Interactive rows receive a restrained hover and focus treatment that communicates navigation without changing the page's established appearance. Row activation supports Enter and Space, and the Back and file-toggle controls remain native buttons.

## Verification

- Add focused component tests for opening `.out`, falling back to `.err`, toggling files, returning to the same list page, and loading/error states where the existing frontend test setup supports them.
- Run the frontend type/build checks.
- Verify the interaction in the running `/app/fasrc` page when the local server and FASRC connection are available.
