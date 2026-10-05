export type LogKind = "out" | "err";

export type RunLogFiles = {
  out_path?: string | null;
  err_path?: string | null;
  missing?: boolean;
};

export type RunWithTaskLogs = RunLogFiles & {
  tasks?: RunLogFiles[] | null;
};

export function preferredLogKind(files: RunLogFiles): LogKind | null {
  // `missing` means the remote directory scan did not see the files.  DB rows
  // still carry their canonical paths and (as in the removed classic UI) the
  // log endpoint makes the authoritative read.  Treat paths as usable here
  // too: scans can be truncated, delayed, or race a newly-created SLURM
  // output file.
  if (files.out_path) return "out";
  if (files.err_path) return "err";
  return null;
}

export function logPath(files: RunLogFiles, kind: LogKind): string | null {
  return kind === "out" ? files.out_path ?? null : files.err_path ?? null;
}

export function hasRunLogs(run: RunWithTaskLogs): boolean {
  if (preferredLogKind(run)) return true;
  return (run.tasks ?? []).some((task) => preferredLogKind(task) != null);
}

export function buildLogPageUrl(path: string, page: number, pageSize: number): string {
  const query = new URLSearchParams({
    path,
    page: String(page),
    page_size: String(pageSize),
  });
  return `/api/fasrc/runs/log?${query.toString()}`;
}

/** SLURM's `%A`/`%a` filename tokens for one array task. */
export function expandArrayPath(path: string | null | undefined, parent: string, index: number): string | null {
  if (!path) return null;
  return String(path).split("%A").join(parent).split("%a").join(String(index));
}

export type LedgerLogRow = {
  jobid: string; label?: string; state?: string; state_display?: string;
  log_path?: string; err_path?: string; params?: Record<string, unknown>; params_json?: string;
};

export type LogTarget = RunLogFiles & {
  name: string; jobid: string; label?: string | null; state?: string | null;
  tasks?: (RunLogFiles & { index: number; member: string; jobid: string; name: string })[];
};

/** The log files of a job-ledger row (history): one `.out`/`.err` pair, or
 *  one per array task (`%A_%a` expanded, named by member). */
export function logTargetFromRow(row: LedgerLogRow): LogTarget {
  let params: Record<string, unknown> = row.params ?? {};
  if (!row.params && row.params_json) {
    try { params = JSON.parse(row.params_json) ?? {}; } catch { params = {}; }
  }
  const count = Math.max(1, Number(params.array_count ?? 1) || 1);
  const base = { jobid: String(row.jobid), label: row.label ?? null, state: row.state_display ?? row.state ?? null };
  const stem = (p: string | null | undefined) => (p ? p.split("/").pop()!.replace(/\.(out|err)$/, "") : "");
  if (count <= 1) {
    return { ...base, name: stem(row.log_path) || String(row.jobid), out_path: row.log_path || null, err_path: row.err_path || null };
  }
  const names = String((params.mode === "continue" ? params.members : params.member_names) ?? "")
    .split(",").map((s) => s.trim()).filter(Boolean);
  const tasks = Array.from({ length: count }, (_, i) => {
    const out = expandArrayPath(row.log_path, String(row.jobid), i);
    return {
      index: i, member: names[i] ?? `task ${i}`, jobid: `${row.jobid}_${i}`, name: stem(out),
      out_path: out, err_path: expandArrayPath(row.err_path, String(row.jobid), i),
    };
  });
  return { ...base, name: stem(row.log_path) || String(row.jobid), out_path: null, err_path: null, tasks };
}
