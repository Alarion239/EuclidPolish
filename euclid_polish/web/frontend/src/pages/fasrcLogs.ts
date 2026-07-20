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
  // still carry their canonical paths and the classic UI lets the log endpoint
  // make the authoritative read.  Treat paths as usable here too: scans can be
  // truncated, delayed, or race a newly-created SLURM output file.
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
