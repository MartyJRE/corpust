// Collocation-by-distance — a heatmap of where each collocate sits
// relative to the node. Rows are the busiest collocates, columns are the
// slot offsets (−5 … −1 │ node │ +1 … +5); cell intensity = how often the
// word lands at that distance. Reveals positional preference at a glance:
// e.g. `freeze` lighting up at −1, `assets` at +1.

import { useEffect, useMemo, useState } from "react";
import {
  type DistanceRow,
  hasLiveData,
  runCollocateDistance,
} from "@/lib/tauri";
import type { CorpusMeta, QueryLayer } from "@/types";

export interface CollocationDistanceProps {
  corpus: CorpusMeta;
  term: string;
  layer: QueryLayer;
}

const WINDOW = 5;
const LIMIT = 25;

export function CollocationDistance({ corpus, term, layer }: CollocationDistanceProps) {
  const isLive = hasLiveData(corpus.id);
  const [offsets, setOffsets] = useState<number[] | null>(null);
  const [rows, setRows] = useState<DistanceRow[] | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setRows(null);
    setOffsets(null);
    if (!isLive || !term.trim()) return;
    let cancelled = false;
    setLoading(true);
    runCollocateDistance({
      corpusId: corpus.id,
      term: term.trim(),
      layer,
      leftWindow: WINDOW,
      rightWindow: WINDOW,
      limit: LIMIT,
    })
      .then((r) => {
        if (cancelled) return;
        setOffsets(r.offsets);
        setRows(r.rows);
      })
      .catch((e) => {
        console.error("runCollocateDistance failed:", e);
        if (!cancelled) {
          setOffsets([]);
          setRows([]);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [corpus.id, term, layer, isLive]);

  const maxCount = useMemo(
    () => Math.max(1, ...(rows ?? []).flatMap((r) => r.counts)),
    [rows],
  );

  // Columns: left offsets, the node, then right offsets.
  const cols = useMemo(() => {
    const os = offsets ?? [];
    const left = os.filter((o) => o < 0);
    const right = os.filter((o) => o > 0);
    return { left, right };
  }, [offsets]);

  const cell = (count: number) => {
    const pct = (count / maxCount) * 100;
    return {
      background: count === 0 ? "transparent" : `color-mix(in oklch, var(--accent) ${Math.max(8, pct)}%, transparent)`,
    };
  };

  const empty = !loading && isLive && rows != null && rows.length === 0;

  return (
    <div className="cx-coll-wrap">
      <div className="cx-coll-main">
        <div className="cx-coll-head">
          <h2 className="cx-coll-title">
            distance profile · <span className="kw">{term}</span>{" "}
            <span style={{ color: "var(--fg-muted)", fontSize: 14 }}>· {corpus.name}</span>
          </h2>
          <div className="cx-coll-controls">
            <span>±{WINDOW} tokens · top {LIMIT}</span>
          </div>
        </div>

        <div className="cx-coll-graph" style={{ overflow: "auto", height: "clamp(460px, 60vh, 780px)", padding: 16 }}>
          {!isLive ? (
            <div className="cx-dist-msg">distance profiles are computed on built corpora</div>
          ) : loading ? (
            <div className="cx-loading-row" style={{ justifyContent: "center", height: "100%" }}>
              <span className="cx-spinner" /> computing distances…
            </div>
          ) : empty ? (
            <div className="cx-dist-msg">
              {layer === "pos"
                ? `no matches for “${term}” on the POS layer — search a tag (NN, VB, JJ…)`
                : `no collocates for “${term}” on this layer`}
            </div>
          ) : rows && offsets ? (
            <table className="cx-dist-table">
              <thead>
                <tr>
                  <th className="cx-dist-word" />
                  {cols.left.map((o) => (
                    <th key={o} className="cx-dist-col">
                      {o}
                    </th>
                  ))}
                  <th className="cx-dist-node">{term}</th>
                  {cols.right.map((o) => (
                    <th key={o} className="cx-dist-col">
                      +{o}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.word}>
                    <td className="cx-dist-word">
                      {r.word} <span className="cx-dist-total">{r.total}</span>
                    </td>
                    {cols.left.map((o) => {
                      const i = offsets.indexOf(o);
                      const c = r.counts[i] ?? 0;
                      return (
                        <td key={o} className="cx-dist-cell" style={cell(c)} title={`${r.word} at ${o}: ${c}`}>
                          {c > 0 ? c : ""}
                        </td>
                      );
                    })}
                    <td className="cx-dist-nodecell" />
                    {cols.right.map((o) => {
                      const i = offsets.indexOf(o);
                      const c = r.counts[i] ?? 0;
                      return (
                        <td key={o} className="cx-dist-cell" style={cell(c)} title={`${r.word} at +${o}: ${c}`}>
                          {c > 0 ? c : ""}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </div>

        <div className="cx-coll-legend">
          <span className="cx-coll-leg-note">
            ← left of node · right of node → · cell shade ∝ count at that slot
          </span>
        </div>
      </div>
    </div>
  );
}
