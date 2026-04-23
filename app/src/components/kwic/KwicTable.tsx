import { ArrowDown, ArrowUp, Download } from "lucide-react";
import { useMemo } from "react";
import type { KwicHit, KwicResult, QueryLayer, SortDir, SortMode } from "@/types";
import { formatDuration } from "@/lib/utils";

export interface KwicTableProps {
  result: KwicResult | null;
  loading?: boolean;
  layer: QueryLayer;
  sortMode: SortMode;
  sortDir: SortDir;
  /** Called when the user clicks a sort header. Same column
   *  toggles direction; different column resets to the default
   *  (asc). The caller is expected to own both values. */
  onSort: (mode: SortMode) => void;
  selected: KwicHit | null;
  onSelect: (h: KwicHit) => void;
}

function sortHits(hits: KwicHit[], mode: SortMode, dir: SortDir): KwicHit[] {
  const h = [...hits];
  const tail = (s: string, n: number) =>
    (s || "").split(/\s+/).slice(-n).join(" ").toLowerCase();
  const head = (s: string, n: number) =>
    (s || "").split(/\s+/).slice(0, n).join(" ").toLowerCase();
  let cmp: (a: KwicHit, b: KwicHit) => number;
  if (mode === "left1") cmp = (a, b) => tail(a.left, 1).localeCompare(tail(b.left, 1));
  else if (mode === "right1") cmp = (a, b) => head(a.right, 1).localeCompare(head(b.right, 1));
  else cmp = (a, b) => a.docId.localeCompare(b.docId);
  h.sort(cmp);
  if (dir === "desc") h.reverse();
  return h;
}

export function KwicTable({
  result,
  loading,
  layer,
  sortMode,
  sortDir,
  onSort,
  selected,
  onSelect,
}: KwicTableProps) {
  const sortedHits = useMemo(
    () => (result ? sortHits(result.hits, sortMode, sortDir) : []),
    [result, sortMode, sortDir],
  );

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
        {(
          [
            ["left1", "left-1"],
            ["right1", "right+1"],
            ["doc", "doc"],
          ] as const
        ).map(([mode, label]) => {
          const active = sortMode === mode;
          return (
            <button
              key={mode}
              type="button"
              className={`cx-sort-btn ${active ? "is-on" : ""}`}
              onClick={() => onSort(mode)}
              title={active ? `click to toggle ${sortDir === "asc" ? "asc→desc" : "desc→asc"}` : `sort by ${label}`}
            >
              {label}{" "}
              <span className="cx-sort-arrow" style={{ opacity: active ? 1 : 0.35 }}>
                {active && sortDir === "desc" ? <ArrowDown size={10} /> : <ArrowUp size={10} />}
              </span>
            </button>
          );
        })}
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
