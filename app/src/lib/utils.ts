import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Pretty-print a duration in microseconds / ms / seconds. */
export function formatDuration(ms: number): string {
  if (ms < 1) return `${Math.round(ms * 1000)}µs`;
  if (ms < 1000) return `${ms.toFixed(1)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

/** Format a token count with thin-space group separators. */
export function formatNumber(n: number): string {
  return n.toLocaleString("en-US").replace(/,/g, " ");
}
