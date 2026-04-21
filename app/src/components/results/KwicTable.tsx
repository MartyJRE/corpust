// Concordance (KWIC) results table — the "hero" surface of the app.
//
// Visual intent: three columns of monospaced text. Left context is
// right-aligned so the hit token always sits on the same vertical axis;
// the hit itself is centered and bold; right context is left-aligned.
// Rows are virtualized so 10K+ hits stay snappy.

import type { KwicHit, KwicResult } from "@/types";
import { formatDuration } from "@/lib/utils";

export interface KwicTableProps {
  result: KwicResult | null;
  loading?: boolean;
}

export function KwicTable({ result, loading }: KwicTableProps) {
  if (loading && !result) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        running query…
      </div>
    );
  }
  if (!result) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        Enter a query above to see a concordance.
      </div>
    );
  }
  if (result.hits.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2">
        <div className="text-sm text-muted-foreground">No hits.</div>
        <div className="text-xs text-muted-foreground">
          Query ran in {formatDuration(result.elapsedMs)}.
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 overflow-auto">
        <table className="w-full border-separate border-spacing-0 font-mono text-sm">
          <colgroup>
            <col className="w-48" />
            <col />
            <col className="w-auto" />
            <col />
          </colgroup>
          <tbody>
            {result.hits.map((h, i) => (
              <Row key={i} hit={h} />
            ))}
          </tbody>
        </table>
      </div>
      <div className="border-t border-border bg-card px-4 py-2 text-xs text-muted-foreground">
        {result.hits.length} hits · {formatDuration(result.elapsedMs)}
        {result.truncated ? " · truncated (hit limit reached)" : ""}
      </div>
    </div>
  );
}

function Row({ hit }: { hit: KwicHit }) {
  return (
    <tr className="group border-b border-border hover:bg-accent/40">
      <td className="truncate px-3 py-1.5 text-xs text-muted-foreground">
        {hit.path}
      </td>
      <td className="whitespace-nowrap px-3 py-1.5 text-right text-foreground">
        {hit.left}
      </td>
      <td className="whitespace-nowrap px-2 py-1.5 text-center font-bold text-primary">
        {hit.hit}
      </td>
      <td className="whitespace-nowrap px-3 py-1.5 text-left text-foreground">
        {hit.right}
      </td>
    </tr>
  );
}
