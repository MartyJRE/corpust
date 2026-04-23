// Collocation scatter — a proper SVG chart.
//
// Semantics
//   x = left/right preference ratio: (rightCount − leftCount) / total.
//       −1 = purely pre-node, 0 = balanced, +1 = purely post-node.
//       The node itself sits at x=0 (dashed amber axis).
//   y = association score (logDice | MI | z-score), user-selectable.
//   radius = sqrt(total frequency) — bigger dot = more occurrences.
//   color = POS family (noun amber / verb teal / adj violet / adv gold /
//           function grey).
//
// A pure-SVG scatter beats absolute-positioned divs because the chart
// becomes visually legible — scale, gridlines, hit-state highlighting,
// smooth metric transitions. All handled with stdlib + a little math;
// no d3 runtime in the bundle.

import { useMemo, useState } from "react";
import { COLLOCATIONS } from "@/data";
import type { CollMetric, Collocate, CorpusMeta } from "@/types";

const POS_FAMILY_COLOR = {
  noun: "var(--layer-word)",
  verb: "var(--layer-lemma)",
  adj: "var(--layer-pos)",
  adv: "var(--warn)",
  function: "var(--fg-subtle)",
} as const;

type PosFamily = keyof typeof POS_FAMILY_COLOR;

function posFamily(tag: string): PosFamily {
  if (tag.startsWith("NN")) return "noun";
  if (tag.startsWith("VB") || tag.startsWith("MD")) return "verb";
  if (tag.startsWith("JJ")) return "adj";
  if (tag.startsWith("RB")) return "adv";
  return "function";
}

function prefRatio(c: Collocate): number {
  if (c.total === 0) return 0;
  return (c.rightCount - c.leftCount) / c.total;
}

function niceTicks(max: number, count = 5): number[] {
  if (max <= 0) return [0];
  const rawStep = max / (count - 1);
  const pow = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const step = Math.ceil(rawStep / pow) * pow;
  const ticks: number[] = [];
  for (let v = 0; v <= max * 1.05; v += step) ticks.push(v);
  return ticks;
}

const W = 760;
const H = 380;
const M = { top: 28, right: 32, bottom: 56, left: 56 };
const PW = W - M.left - M.right;
const PH = H - M.top - M.bottom;

export interface CollocationsViewProps {
  corpus: CorpusMeta | null;
  term: string;
  /** Real collocates from the backend. When null, falls back to the
   *  fixture `COLLOCATIONS` so the view still looks populated in
   *  demos / non-Tauri preview. */
  data?: Collocate[] | null;
  /** Tokens on the left of the node to consider (0 = skip). Lifted
   *  to App so the backend refetch uses the same values the UI shows. */
  leftWindow?: number;
  rightWindow?: number;
  onWindowChange?: (left: number, right: number) => void;
}

const WINDOW_CHOICES = [0, 1, 2, 3, 5, 7, 10] as const;

export function CollocationsView({
  term,
  data: dataProp,
  leftWindow = 5,
  rightWindow = 5,
  onWindowChange,
}: CollocationsViewProps) {
  const [metric, setMetric] = useState<CollMetric>("logDice");
  const [hover, setHover] = useState<string | null>(null);

  const data = useMemo(
    () => (dataProp && dataProp.length > 0 ? dataProp : COLLOCATIONS),
    [dataProp],
  );
  const maxScore = useMemo(() => Math.max(1e-6, ...data.map((d) => d[metric])), [data, metric]);
  const maxTotal = useMemo(() => Math.max(1, ...data.map((d) => d.total)), [data]);

  const xFor = (pref: number) => M.left + ((pref + 1) / 2) * PW;
  const yFor = (score: number) => M.top + PH - (score / (maxScore * 1.05)) * PH;
  const rFor = (total: number) => 4 + (Math.sqrt(total) / Math.sqrt(maxTotal)) * 12;
  const colorOf = (pos: string) => POS_FAMILY_COLOR[posFamily(pos)];

  const yTicks = niceTicks(maxScore, 5);
  const xTicks = [-1, -0.5, 0, 0.5, 1];
  const metricLabel = metric === "logDice" ? "log-Dice" : metric === "mi" ? "MI" : "z-score";

  return (
    <div className="cx-coll-wrap">
      <div className="cx-coll-main">
        <div className="cx-coll-head">
          <h2 className="cx-coll-title">
            collocates of <span className="kw">{term}</span>
          </h2>
          <div className="cx-coll-controls">
            <span>L</span>
            <div className="cx-coll-segbtn">
              {WINDOW_CHOICES.map((w) => (
                <button
                  key={`l-${w}`}
                  type="button"
                  className={w === leftWindow ? "is-on" : ""}
                  onClick={() => onWindowChange?.(w, rightWindow)}
                  disabled={!onWindowChange}
                  title={w === 0 ? "skip left context" : `±${w} tokens left`}
                >
                  {w}
                </button>
              ))}
            </div>
            <span style={{ marginLeft: 12 }}>R</span>
            <div className="cx-coll-segbtn">
              {WINDOW_CHOICES.map((w) => (
                <button
                  key={`r-${w}`}
                  type="button"
                  className={w === rightWindow ? "is-on" : ""}
                  onClick={() => onWindowChange?.(leftWindow, w)}
                  disabled={!onWindowChange}
                  title={w === 0 ? "skip right context" : `±${w} tokens right`}
                >
                  {w}
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

        <div className="cx-coll-graph cx-coll-svg-wrap">
          <svg
            viewBox={`0 0 ${W} ${H}`}
            preserveAspectRatio="xMidYMid meet"
            style={{ width: "100%", height: "100%", display: "block" }}
          >
            {/* Y gridlines + labels */}
            {yTicks.map((t) => (
              <g key={`y-${t}`}>
                <line
                  x1={M.left}
                  x2={W - M.right}
                  y1={yFor(t)}
                  y2={yFor(t)}
                  stroke="var(--border)"
                  strokeWidth={1}
                  opacity={t === 0 ? 0.6 : 0.35}
                />
                <text
                  x={M.left - 10}
                  y={yFor(t)}
                  textAnchor="end"
                  dominantBaseline="middle"
                  style={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--fg-subtle)" }}
                >
                  {t.toFixed(t > 1 ? 0 : 1)}
                </text>
              </g>
            ))}

            {/* Y axis title */}
            <text
              x={18}
              y={M.top + PH / 2}
              textAnchor="middle"
              dominantBaseline="middle"
              transform={`rotate(-90 18 ${M.top + PH / 2})`}
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                fill: "var(--fg-subtle)",
                textTransform: "uppercase",
                letterSpacing: "0.08em",
              }}
            >
              {metricLabel}
            </text>

            {/* Vertical center (node) line */}
            <line
              x1={xFor(0)}
              x2={xFor(0)}
              y1={M.top - 4}
              y2={H - M.bottom + 4}
              stroke="var(--accent)"
              strokeWidth={1}
              strokeDasharray="3 4"
              opacity={0.4}
            />

            {/* X axis baseline + ticks */}
            <line
              x1={M.left}
              x2={W - M.right}
              y1={H - M.bottom}
              y2={H - M.bottom}
              stroke="var(--border)"
            />
            {xTicks.map((t) => (
              <g key={`x-${t}`}>
                <line
                  x1={xFor(t)}
                  x2={xFor(t)}
                  y1={H - M.bottom}
                  y2={H - M.bottom + 4}
                  stroke="var(--border)"
                />
                <text
                  x={xFor(t)}
                  y={H - M.bottom + 18}
                  textAnchor="middle"
                  style={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--fg-subtle)" }}
                >
                  {t === 0 ? "" : t > 0 ? `+${t}` : `${t}`}
                </text>
              </g>
            ))}

            {/* Axis side labels */}
            <text
              x={M.left}
              y={H - 8}
              textAnchor="start"
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                fill: "var(--fg-subtle)",
                textTransform: "uppercase",
                letterSpacing: "0.08em",
              }}
            >
              ← left of node
            </text>
            <text
              x={W - M.right}
              y={H - 8}
              textAnchor="end"
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                fill: "var(--fg-subtle)",
                textTransform: "uppercase",
                letterSpacing: "0.08em",
              }}
            >
              right of node →
            </text>

            {/* Node label at x=0 above the plot */}
            <text
              x={xFor(0)}
              y={M.top - 10}
              textAnchor="middle"
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                fill: "var(--accent)",
                fontWeight: 600,
                letterSpacing: "0.02em",
              }}
            >
              {term}
            </text>

            {/* Data points */}
            {data.map((c) => {
              const cx = xFor(prefRatio(c));
              const cy = yFor(c[metric]);
              const r = rFor(c.total);
              const color = colorOf(c.pos);
              const labelRight = prefRatio(c) <= 0;
              const isHover = hover === c.word;
              const dimmed = hover !== null && !isHover;
              return (
                <g
                  key={c.word}
                  transform={`translate(${cx}, ${cy})`}
                  style={{
                    cursor: "pointer",
                    transition: "transform 280ms cubic-bezier(.4,.0,.2,1), opacity 180ms",
                    opacity: dimmed ? 0.28 : 1,
                  }}
                  onMouseEnter={() => setHover(c.word)}
                  onMouseLeave={() => setHover(null)}
                >
                  <circle
                    r={r}
                    fill={color}
                    fillOpacity={isHover ? 0.42 : 0.22}
                    stroke={color}
                    strokeWidth={isHover ? 2 : 1.2}
                    style={{ transition: "r 220ms ease, fill-opacity 180ms, stroke-width 180ms" }}
                  />
                  <text
                    x={labelRight ? r + 6 : -(r + 6)}
                    y={4}
                    textAnchor={labelRight ? "start" : "end"}
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: isHover ? 13 : 12,
                      fill: "var(--fg)",
                      fontWeight: isHover ? 600 : 400,
                      pointerEvents: "none",
                      transition: "font-size 160ms",
                    }}
                  >
                    {c.word}
                  </text>
                  {isHover && (
                    <g transform={`translate(${labelRight ? r + 6 : -(r + 6)}, 18)`}>
                      <text
                        textAnchor={labelRight ? "start" : "end"}
                        style={{
                          fontFamily: "var(--font-mono)",
                          fontSize: 10,
                          fill: "var(--fg-muted)",
                          pointerEvents: "none",
                        }}
                      >
                        {c.pos} · {metricLabel} {c[metric].toFixed(1)} · {c.total.toLocaleString()}×
                      </text>
                    </g>
                  )}
                </g>
              );
            })}
          </svg>
        </div>

        <div className="cx-coll-legend">
          {(
            [
              ["noun", POS_FAMILY_COLOR.noun],
              ["verb", POS_FAMILY_COLOR.verb],
              ["adjective", POS_FAMILY_COLOR.adj],
              ["adverb", POS_FAMILY_COLOR.adv],
              ["function", POS_FAMILY_COLOR.function],
            ] as const
          ).map(([name, color]) => (
            <span key={name} className="cx-coll-leg-item">
              <span className="cx-coll-leg-dot" style={{ background: color, borderColor: color }} />
              {name}
            </span>
          ))}
          <span className="cx-coll-leg-sep">·</span>
          <span className="cx-coll-leg-note">dot size ∝ total frequency</span>
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
              <tr
                key={c.word}
                onMouseEnter={() => setHover(c.word)}
                onMouseLeave={() => setHover(null)}
                style={hover === c.word ? { background: "color-mix(in oklch, var(--bg-accent) 40%, transparent)" } : undefined}
              >
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
