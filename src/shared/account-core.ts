export function normalizeHandle(value: string | null | undefined): string {
  return (value ?? "").trim().replace(/^\/+/, "").replace(/^@+/, "").toLowerCase();
}
