// Helpers for the document-metadata query filter: emptiness checks,
// normalization (trim + drop empty dimensions), and the chip list the
// QueryBar renders. The same `DocFilter` shape travels to the backend.

import type { DocFilter } from "@/types";

export function isFilterEmpty(f: DocFilter): boolean {
  return (
    f.yearMin == null &&
    f.yearMax == null &&
    !f.author?.trim() &&
    !f.path?.trim()
  );
}

/** Trim strings and drop empty dimensions. Returns `undefined` when
 *  nothing is set, so query requests omit `filter` entirely (the backend
 *  treats an absent filter as "whole corpus"). */
export function normalizeFilter(f: DocFilter): DocFilter | undefined {
  const out: DocFilter = {};
  if (f.yearMin != null) out.yearMin = f.yearMin;
  if (f.yearMax != null) out.yearMax = f.yearMax;
  if (f.author?.trim()) out.author = f.author.trim();
  if (f.path?.trim()) out.path = f.path.trim();
  return isFilterEmpty(out) ? undefined : out;
}

export type FilterDimension = "year" | "author" | "path";

export interface FilterChip {
  key: FilterDimension;
  label: string;
}

/** One chip per active dimension, for display in the query bar. */
export function filterChips(f: DocFilter): FilterChip[] {
  const chips: FilterChip[] = [];
  if (f.yearMin != null || f.yearMax != null) {
    const lo = f.yearMin != null ? String(f.yearMin) : "…";
    const hi = f.yearMax != null ? String(f.yearMax) : "…";
    chips.push({ key: "year", label: `year: ${lo}–${hi}` });
  }
  if (f.author?.trim()) chips.push({ key: "author", label: `author: ${f.author.trim()}` });
  if (f.path?.trim()) chips.push({ key: "path", label: `path: ${f.path.trim()}` });
  return chips;
}

/** Client-side mirror of the backend predicate, for views that join a
 *  document list locally (e.g. frequency-over-time, which needs the
 *  subcorpus token denominator). Keep in sync with `DocFilter::matches`
 *  in corpust-index. */
export function docMatchesFilter(
  doc: { path: string; author: string | null; year: number | null },
  f?: DocFilter,
): boolean {
  if (!f) return true;
  if (f.yearMin != null || f.yearMax != null) {
    if (doc.year == null) return false;
    if (f.yearMin != null && doc.year < f.yearMin) return false;
    if (f.yearMax != null && doc.year > f.yearMax) return false;
  }
  if (f.author?.trim() && !doc.author?.toLowerCase().includes(f.author.trim().toLowerCase())) {
    return false;
  }
  if (f.path?.trim() && !doc.path.toLowerCase().includes(f.path.trim().toLowerCase())) {
    return false;
  }
  return true;
}

/** Drop one dimension from a filter (chip ✕). */
export function clearDimension(f: DocFilter, key: FilterDimension): DocFilter {
  const next = { ...f };
  if (key === "year") {
    delete next.yearMin;
    delete next.yearMax;
  } else {
    delete next[key];
  }
  return next;
}
