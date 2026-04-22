// Novel visualization: horizontal axis = token distance from the node
// (−window … +window). Vertical = association score (logDice / MI / z).
// Stronger collocates float higher; left collocates on the left of the
// node, right on the right.

import { useState } from "react";
import { COLLOCATIONS } from "@/data";
import type { CollMetric, CorpusMeta } from "@/types";

export interface CollocationsViewProps {
  corpus: CorpusMeta | null;
  term: string;
}

export function CollocationsView({ term }: CollocationsViewProps) {
  const [metric, setMetric] = useState<CollMetric>("logDice");
  const [win, setWin] = useState<3 | 5 | 10>(5);

  const data = COLLOCATIONS;
  const maxY = Math.max(...data.map((d) => d[metric]));

  const placed = data.map((d) => {
    const x = ((d.dist + win) / (win * 2)) * 100;
    const y = 100 - (d[metric] / (maxY * 1.05)) * 92;
    return { ...d, x, y };
  });

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
                <button key={w} type="button" className={w === win ? "is-on" : ""} onClick={() => setWin(w)}>
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
          {[-5, -3, -1, 1, 3, 5].map((t) => {
            const x = ((t + win) / (win * 2)) * 100;
            return (
              <span key={t}>
                <div className="cx-coll-tick" style={{ left: `${x}%` }} />
                <div className="cx-coll-tick-label" style={{ left: `${x}%` }}>
                  {t > 0 ? `+${t}` : t}
                </div>
              </span>
            );
          })}
          <div className="cx-coll-axis-label left">left of node</div>
          <div className="cx-coll-axis-label right">right of node</div>
          <div className="cx-coll-center">{term}</div>
          {placed.map((p) => (
            <div
              key={p.word}
              className={`cx-coll-node ${p[metric] > maxY * 0.6 ? "strong" : ""}`}
              style={{
                left: `calc(${p.x}% - 24px)`,
                top: `calc(${p.y}% - 10px)`,
                fontSize: Math.min(16, 10 + p[metric] * 0.5),
              }}
              title={`${p.word} · logDice ${p.logDice} · MI ${p.mi}`}
            >
              {p.word}
            </div>
          ))}
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
                  <span className="cx-meter" style={{ width: Math.min(120, c[metric] * 12) }} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
