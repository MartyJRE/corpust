// File export for the concordance and the corpus document list. The rows
// are already in memory, so the text is built client-side and written
// out — via the native save dialog inside Tauri, or a browser download in
// the vite-only preview.

import type { CorpusMeta, KwicResult, QueryLayer } from "@/types";
import { inTauri, writeTextFile } from "@/lib/tauri";

/** RFC-4180-ish field escaping: quote when the value carries a comma,
 *  quote, CR or LF, doubling any embedded quotes. */
function csvField(value: string | number): string {
  const s = String(value);
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function toCsv(headers: string[], rows: (string | number)[][]): string {
  return [headers, ...rows].map((r) => r.map(csvField).join(",")).join("\r\n") + "\r\n";
}

/** A filesystem-safe slug for default export filenames. */
export function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "corpus";
}

export function concordanceCsv(result: KwicResult): string {
  return toCsv(
    ["n", "doc", "position", "left", "node", "right"],
    result.hits.map((h, i) => [i + 1, h.docId, h.pos, h.left, h.hit, h.right]),
  );
}

export function concordanceJson(
  result: KwicResult,
  corpus: CorpusMeta | null,
  term: string,
  layer: QueryLayer,
): string {
  return JSON.stringify(
    {
      corpus: corpus?.name ?? null,
      term,
      layer,
      total: result.total,
      hits: result.hits.map((h) => ({
        doc: h.docId,
        position: h.pos,
        left: h.left,
        node: h.hit,
        right: h.right,
        ...(h.lemma ? { lemma: h.lemma } : {}),
        ...(h.pos_tag ? { posTag: h.pos_tag } : {}),
      })),
    },
    null,
    2,
  );
}

export interface DocExportRow {
  file: string;
  title: string | null;
  author: string | null;
  year: number | null;
  tokens: number;
}

export function documentsCsv(rows: DocExportRow[]): string {
  return toCsv(
    ["file", "title", "author", "year", "tokens"],
    rows.map((d) => [d.file, d.title ?? "", d.author ?? "", d.year ?? "", d.tokens]),
  );
}

/** Write text to a user-chosen path (Tauri save dialog) or trigger a
 *  browser download (vite preview). No-ops if the user cancels. */
export async function saveText(suggestedName: string, contents: string, mime: string): Promise<void> {
  if (inTauri()) {
    const { save } = await import("@tauri-apps/plugin-dialog");
    const ext = suggestedName.includes(".") ? suggestedName.split(".").pop()! : "";
    const path = await save({
      defaultPath: suggestedName,
      filters: ext ? [{ name: ext.toUpperCase(), extensions: [ext] }] : undefined,
    });
    if (!path) return;
    await writeTextFile(path, contents);
    return;
  }
  const blob = new Blob([contents], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = suggestedName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
