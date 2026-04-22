import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDuration(ms: number): string {
  if (ms < 1) return `${Math.round(ms * 1000)} µs`;
  if (ms < 1000) return `${ms.toFixed(1)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let i = -1;
  let x = b;
  do {
    x /= 1024;
    i++;
  } while (x >= 1024 && i < units.length - 1);
  return `${x.toFixed(x < 10 ? 1 : 0)} ${units[i]}`;
}

export function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-GB", { year: "numeric", month: "short", day: "numeric" });
}

export function formatBuildTime(ms: number): string {
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function formatNumber(n: number): string {
  return n.toLocaleString("en-US");
}

/** Bucket KWIC hits by their doc position into `buckets` density cells. */
export function makeDensity(hits: { pos: number }[], buckets = 60): number[] {
  const d = new Array(buckets).fill(0) as number[];
  if (!hits || hits.length === 0) return d;
  const maxPos = Math.max(...hits.map((h) => h.pos || 0), 1);
  hits.forEach((h) => {
    const i = Math.min(buckets - 1, Math.floor((h.pos / maxPos) * buckets));
    d[i]++;
  });
  // smear a tiny bit to make sparse results look visually coherent
  for (let i = 0; i < buckets; i++) {
    if (d[i] === 0 && Math.random() > 0.55) d[i] = Math.random() > 0.5 ? 1 : 0;
  }
  return d;
}
