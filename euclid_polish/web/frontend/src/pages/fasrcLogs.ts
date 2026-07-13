export type LogKind = "out" | "err";

export type RunLogFiles = {
  out_path?: string | null;
  err_path?: string | null;
  missing?: boolean;
};

export function preferredLogKind(files: RunLogFiles): LogKind | null {
  if (files.missing) return null;
  if (files.out_path) return "out";
  if (files.err_path) return "err";
  return null;
}

export function logPath(files: RunLogFiles, kind: LogKind): string | null {
  return kind === "out" ? files.out_path ?? null : files.err_path ?? null;
}

export function buildLogPageUrl(path: string, page: number, pageSize: number): string {
  const query = new URLSearchParams({
    path,
    page: String(page),
    page_size: String(pageSize),
  });
  return `/api/fasrc/runs/log?${query.toString()}`;
}
