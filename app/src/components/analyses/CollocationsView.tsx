// Novel visualization: horizontal axis = token distance from the node
// (−window … +window). Vertical = association-score ladder within each
// distance column (top scorer at the top; step down by rank).
//
// The original scatter-by-score design collapsed to overlapping labels
// because real-world collocate dist values cluster at ±1. We sort
// per-column by score, stack vertically with a consistent step, and
// add a small alternating x-jitter so labels don't fall on a perfectly
// straight line.

import { Fragment, useState } from "react";
import { COLLOCATIONS } from "@/data";
import type { CollMetric, Collocate, CorpusMeta } from "@/types";

export interface CollocationsViewProps {
  corpus: CorpusMeta | null;
  term: string;
}

interface PlacedNode {
  c: Collocate;
  x: number;
  y: number;
}

function layoutNodes(data: Collocate[], metric: CollMetric, win: number): PlacedNode[] {
  const byDist = new Map<number, Collocate[]>();
  for (const d of data) {
    const arr = byDist.get(d.dist) ?? [];
    arr.push(d);
    byDist.set(d.dist, arr);
  }
  // Sort each group by score descending so top scorers sit at the top.
  for (const arr of byDist.values()) {
    arr.sort((a, b) => b[metric] - a[metric]);
  }

  const placed: PlacedNode[] = [];
  for (const [dist, arr] of byDist.entries()) {
    const baseX = ((dist + win) / (win * 2)) * 100;
    arr.forEach((c, i) => {
      const y = 10 + i * 10; // ladder — 10% stride per rank
      // Tight alternating jitter so equal-rank labels aren't perfectly stacked
      const jitterSign = i % 2 === 0 ? -1 : 1;
      const jitterMag = Math.min(6, 1.5 + i * 0.6);
      const x = Math.max(4, Math.min(96, baseX + jitterSign * jitterMag));
      placed.push({ c, x, y });
    });
  }
  return placed;
}

const TICKS = [-4, -2, 0, 2, 4] as const;

export function CollocationsView({ term }: CollocationsViewProps) {
  const [metric, setMetric] = useState<CollMetric>("logDice");
  const [win, setWin] = useState<3 | 5 | 10>(5);

  const data = COLLOCATIONS;
  const maxScore = Math.max(...data.map((d) => d[metric]));
  const placed = layoutNodes(data, metric, win);

  return (
    <div className="cx-coll-wrap">
      <div className="cx-coll-main">
        <div className="cx-coll-head">
          <h2 className="cx-coll-title">
            collocates of <span className="kw">{term}</span>
          </h2>
          <div className="cx-coll-controls">
            <span>window</span>
            <div className="cx-coll-segbtn">
              {([3, 5, 10] as const).map((w) => (
                <button
                  key={w}
                  type="button"
                  className={w === win ? "is-on" : ""}
                  onClick={() => setWin(w)}
                >
                  ±{w}
                </button>
              ))}
            </div>
            <span style={{ marginLeft: 12 }}>score</span>
            <div className="cx-coll-segbtn">
              {(
                [
                  ["logDice", "logDice"],
                  ["mi", "MI"],
                  ["z", "z-score"],
                ] as const
              ).map(([k, l]) => (
                <button
                  key={k}
                  type="button"
                  className={metric === k ? "is-on" : ""}
                  onClick={() => setMetric(k)}
                >
                  {l}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="cx-coll-graph">
          <div className="cx-coll-axis" />
          {TICKS.map((t) => {
            const x = ((t + win) / (win * 2)) * 100;
            return (
              <Fragment key={t}>
                <div className="cx-coll-tick" style={{ left: `${x}%` }} />
                <div className="cx-coll-tick-label" style={{ left: `${x}%` }}>
                  {t > 0 ? `+${t}` : t}
                </div>
              </Fragment>
            );
          })}
          <div className="cx-coll-axis-label left">left of node</div>
          <div className="cx-coll-axis-label right">right of node</div>
          <div className="cx-coll-center">{term}</div>
          {placed.map((p) => {
            const fs = Math.min(15, 10 + (p.c[metric] / maxScore) * 5);
            return (
              <div
                key={p.c.word}
                className={`cx-coll-node ${p.c[metric] > maxScore * 0.6 ? "strong" : ""}`}
                style={{
                  left: `calc(${p.x}% - 24px)`,
                  top: `${p.y}%`,
                  fontSize: fs,
                }}
                title={`${p.c.word} · logDice ${p.c.logDice} · MI ${p.c.mi} · z ${p.c.z}`}
              >
                {p.c.word}
              </div>
            );
          })}
        </div>

        <table className="cx-coll-table">
          <thead>
            <tr>
              <th>collocate</th>
              <th>pos</th>
              <th className="num">L</th>
              <th className="num">R</th>
              <th className="num">total</th>
              <th className="num">logDice</th>
              <th className="num">MI</th>
              <th className="num">z-score</th>
              <th className="num">strength</th>
            </tr>
          </thead>
          <tbody>
            {data.map((c) => (
              <tr key={c.word}>
                <td className="word">{c.word}</td>
                <td>
                  <span className="cx-layer-chip cx-layer-pos">{c.pos}</span>
                </td>
                <td className="num">{c.leftCount}</td>
                <td className="num">{c.rightCount}</td>
                <td className="num">{c.total}</td>
                <td className="num">{c.logDice.toFixed(2)}</td>
                <td className="num">{c.mi.toFixed(2)}</td>
                <td className="num">{c.z.toFixed(1)}</td>
                <td className="num">
                  <span
                    className="cx-meter"
                    style={{ width: Math.max(4, (c[metric] / maxScore) * 120) }}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
