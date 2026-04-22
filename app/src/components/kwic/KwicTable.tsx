import { Download } from "lucide-react";
import { useMemo } from "react";
import type { KwicHit, KwicResult, QueryLayer, SortMode } from "@/types";
import { formatDuration } from "@/lib/utils";

export interface KwicTableProps {
  result: KwicResult | null;
  loading?: boolean;
  layer: QueryLayer;
  sortMode: SortMode;
  onSort: (s: SortMode) => void;
  selected: KwicHit | null;
  onSelect: (h: KwicHit) => void;
}

function sortHits(hits: KwicHit[], mode: SortMode): KwicHit[] {
  const h = [...hits];
  const tail = (s: string, n: number) =>
    (s || "").split(/\s+/).slice(-n).join(" ").toLowerCase();
  const head = (s: string, n: number) =>
    (s || "").split(/\s+/).slice(0, n).join(" ").toLowerCase();
  if (mode === "left1") h.sort((a, b) => tail(a.left, 1).localeCompare(tail(b.left, 1)));
  if (mode === "right1") h.sort((a, b) => head(a.right, 1).localeCompare(head(b.right, 1)));
  if (mode === "doc") h.sort((a, b) => a.docId.localeCompare(b.docId));
  return h;
}

export function KwicTable({ result, loading, layer, sortMode, onSort, selected, onSelect }: KwicTableProps) {
  const sortedHits = useMemo(() => (result ? sortHits(result.hits, sortMode) : []), [result, sortMode]);

  if (loading && !result) {
    return (
      <div className="cx-results-empty">
        <span>running query…</span>
      </div>
    );
  }
  if (!result) {
    return (
      <div className="cx-results-empty">
        <div>Enter a query above to see a concordance.</div>
        <div className="cx-results-empty-hint">
          <span className="cx-kbd">⌘K</span> for commands
        </div>
      </div>
    );
  }
  if (result.hits.length === 0) {
    return (
      <div className="cx-results-empty">
        <div>No hits.</div>
        <div className="cx-results-empty-sub">query ran in {formatDuration(result.elapsedMs)}</div>
      </div>
    );
  }

  return (
    <div className="cx-kwic-col">
      <div className="cx-kwic-sort-bar">
        <span>sort</span>
        <button
          type="button"
          className={`cx-sort-btn ${sortMode === "left1" ? "is-on" : ""}`}
          onClick={() => onSort("left1")}
        >
          left-1 <span className="cx-sort-arrow">↑</span>
        </button>
        <button
          type="button"
          className={`cx-sort-btn ${sortMode === "right1" ? "is-on" : ""}`}
          onClick={() => onSort("right1")}
        >
          right+1 <span className="cx-sort-arrow">↑</span>
        </button>
        <button
          type="button"
          className={`cx-sort-btn ${sortMode === "doc" ? "is-on" : ""}`}
          onClick={() => onSort("doc")}
        >
          doc <span className="cx-sort-arrow">↑</span>
        </button>
        <span className="sep">·</span>
        <span>{result.hits.length.toLocaleString()} hits</span>
        <span className="sep">·</span>
        <span className="cx-status-time">{formatDuration(result.elapsedMs)}</span>
        <span className="cx-spacer" />
        <button type="button" className="cx-sort-btn">
          <Download size={11} /> export csv
        </button>
      </div>
      <table className="cx-kwic">
        <colgroup>
          <col style={{ width: 38 }} />
          <col style={{ width: 180 }} />
          <col style={{ width: 60 }} />
          <col />
          <col style={{ width: 140 }} />
          <col />
        </colgroup>
        <tbody>
          {sortedHits.map((h, i) => {
            const isSel = selected != null && selected.docId === h.docId && selected.pos === h.pos;
            return (
              <tr
                key={`${h.docId}-${h.pos}`}
                className={`cx-kwic-row ${isSel ? "is-sel" : ""}`}
                onClick={() => onSelect(h)}
              >
                <td className="cx-kwic-td cx-kwic-idx">{String(i + 1).padStart(3, " ")}</td>
                <td className="cx-kwic-td cx-kwic-cpath" title={h.docId}>
                  {h.docId}
                </td>
                <td className="cx-kwic-td cx-kwic-meta">{h.pos.toLocaleString()}</td>
                <td className="cx-kwic-td cx-kwic-left">{h.left}</td>
                <td className={`cx-kwic-td cx-kwic-chit cx-layer-${layer}`}>{h.hit}</td>
                <td className="cx-kwic-td cx-kwic-right">{h.right}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
