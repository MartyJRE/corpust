// Frequency over time — normalized frequency of the query term per year,
// the diachronic "how does usage change" figure. Built client-side by
// joining two existing queries: per-doc hit counts (term_distribution)
// with each doc's heuristic publication year (list_documents). For each
// year we sum hits and tokens across its documents and plot hits per
// million tokens, so a year with more text doesn't look busier just for
// being bigger.

import { useEffect, useMemo, useState } from "react";
import { docMatchesFilter } from "@/lib/filter";
import { hasLiveData, listDocuments, runTermDistribution } from "@/lib/tauri";
import type { CorpusMeta, DocFilter, QueryLayer } from "@/types";

export interface FrequencyOverTimeProps {
  corpus: CorpusMeta;
  term: string;
  layer: QueryLayer;
  /** Active metadata filter (already normalized). Scopes the hit counts
   *  and the per-year token denominator to the same subcorpus. */
  filter?: DocFilter;
}

interface YearPoint {
  year: number;
  hits: number;
  tokens: number;
  per1m: number;
}

const W = 820;
const H = 420;
const M = { top: 28, right: 28, bottom: 44, left: 60 };
const PW = W - M.left - M.right;
const PH = H - M.top - M.bottom;

function niceMax(v: number): number {
  if (v <= 0) return 1;
  const pow = 10 ** Math.floor(Math.log10(v));
  const n = v / pow;
  const step = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
  return step * pow;
}

function yearTicks(min: number, max: number, count = 6): number[] {
  if (min === max) return [min];
  const span = max - min;
  const raw = span / (count - 1);
  const pow = 10 ** Math.floor(Math.log10(raw));
  const step = Math.max(1, Math.ceil(raw / pow) * pow);
  const start = Math.ceil(min / step) * step;
  const out: number[] = [];
  for (let v = start; v <= max + 0.5; v += step) out.push(v);
  return out;
}

export function FrequencyOverTime({ corpus, term, layer, filter }: FrequencyOverTimeProps) {
  const isLive = hasLiveData(corpus.id);
  const [points, setPoints] = useState<YearPoint[] | null>(null);
  const [undated, setUndated] = useState(0);
  const [loading, setLoading] = useState(false);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    setPoints(null);
    setUndated(0);
    if (!isLive || !term.trim()) return;
    let cancelled = false;
    setLoading(true);
    Promise.all([
      listDocuments(corpus.id),
      runTermDistribution({ corpusId: corpus.id, term: term.trim(), layer, buckets: 100, filter }),
    ])
      .then(([allDocs, dist]) => {
        if (cancelled) return;
        // Restrict the per-year token denominator to the same subcorpus
        // the filter scoped the hits to, so per-1M stays accurate.
        const docs = allDocs.filter((d) => docMatchesFilter(d, filter));
        const hitsByDoc = new Map<number, number>();
        for (const d of dist.docCounts) hitsByDoc.set(d.docId, d.hits);

        const tokensByYear = new Map<number, number>();
        const hitsByYear = new Map<number, number>();
        let undatedCount = 0;
        for (const d of docs) {
          if (d.year == null) {
            undatedCount++;
            continue;
          }
          tokensByYear.set(d.year, (tokensByYear.get(d.year) ?? 0) + d.tokenCount);
          hitsByYear.set(d.year, (hitsByYear.get(d.year) ?? 0) + (hitsByDoc.get(d.docId) ?? 0));
        }
        const pts: YearPoint[] = [...tokensByYear.entries()]
          .map(([year, tokens]) => {
            const hits = hitsByYear.get(year) ?? 0;
            return { year, hits, tokens, per1m: tokens > 0 ? (hits / tokens) * 1_000_000 : 0 };
          })
          .sort((a, b) => a.year - b.year);
        setPoints(pts);
        setUndated(undatedCount);
      })
      .catch((e) => {
        console.error("frequency-over-time failed:", e);
        if (!cancelled) setPoints([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [corpus.id, term, layer, isLive, filter]);

  const scale = useMemo(() => {
    const pts = points ?? [];
    if (pts.length === 0) return null;
    const minYear = pts[0].year;
    const maxYear = pts[pts.length - 1].year;
    const yMax = niceMax(Math.max(...pts.map((p) => p.per1m)));
    const xFor = (year: number) =>
      minYear === maxYear ? M.left + PW / 2 : M.left + ((year - minYear) / (maxYear - minYear)) * PW;
    const yFor = (v: number) => M.top + PH - (v / yMax) * PH;
    return { minYear, maxYear, yMax, xFor, yFor };
  }, [points]);

  const empty = !loading && isLive && points != null && points.length === 0;

  return (
    <div className="cx-coll-wrap">
      <div className="cx-coll-main">
        <div className="cx-coll-head">
          <h2 className="cx-coll-title">
            frequency over time · <span className="kw">{term}</span>{" "}
            <span style={{ color: "var(--fg-muted)", fontSize: 14 }}>· {corpus.name}</span>
          </h2>
          <div className="cx-coll-controls">
            <span>per 1M tokens</span>
          </div>
        </div>

        <div className="cx-coll-graph" style={{ height: "clamp(420px, 56vh, 720px)", position: "relative" }}>
          {!isLive ? (
            <div className="cx-dist-msg">frequency-over-time is computed on built corpora</div>
          ) : loading ? (
            <div className="cx-loading-row" style={{ justifyContent: "center", height: "100%" }}>
              <span className="cx-spinner" /> computing trend…
            </div>
          ) : empty ? (
            <div className="cx-dist-msg">
              {!term.trim()
                ? "enter a search term to see its trend over time"
                : undated > 0
                  ? `none of this corpus's ${undated.toLocaleString()} documents have a detected year — rebuild the corpus to extract publication dates`
                  : `no dated occurrences of “${term}” to chart`}
            </div>
          ) : points && scale ? (
            <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" style={{ width: "100%", height: "100%", display: "block" }}>
              {/* Y gridlines + labels */}
              {[0, 0.25, 0.5, 0.75, 1].map((f) => {
                const v = scale.yMax * f;
                return (
                  <g key={f}>
                    <line x1={M.left} x2={W - M.right} y1={scale.yFor(v)} y2={scale.yFor(v)} stroke="var(--border)" opacity={f === 0 ? 0.6 : 0.3} />
                    <text x={M.left - 8} y={scale.yFor(v)} textAnchor="end" dominantBaseline="middle" style={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--fg-subtle)" }}>
                      {v.toFixed(v < 10 ? 1 : 0)}
                    </text>
                  </g>
                );
              })}
              <text x={16} y={M.top + PH / 2} textAnchor="middle" transform={`rotate(-90 16 ${M.top + PH / 2})`} style={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--fg-subtle)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                per 1M
              </text>

              {/* X axis ticks (years) */}
              <line x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} stroke="var(--border)" />
              {yearTicks(scale.minYear, scale.maxYear).map((y) => (
                <text key={y} x={scale.xFor(y)} y={H - M.bottom + 16} textAnchor="middle" style={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--fg-subtle)" }}>
                  {y}
                </text>
              ))}

              {/* Area + line */}
              {points.length > 1 && (
                <path
                  d={`M ${points.map((p) => `${scale.xFor(p.year)} ${scale.yFor(p.per1m)}`).join(" L ")}`}
                  fill="none"
                  stroke="var(--accent)"
                  strokeWidth={1.6}
                />
              )}
              {points.map((p) => {
                const isHover = hover === p.year;
                return (
                  <g key={p.year} onMouseEnter={() => setHover(p.year)} onMouseLeave={() => setHover(null)} style={{ cursor: "pointer" }}>
                    {/* generous transparent hit target */}
                    <rect x={scale.xFor(p.year) - 8} y={M.top} width={16} height={PH} fill="transparent" />
                    <circle cx={scale.xFor(p.year)} cy={scale.yFor(p.per1m)} r={isHover ? 4.5 : 3} fill="var(--accent)" stroke="var(--bg)" strokeWidth={1} />
                    {isHover && (
                      <g transform={`translate(${scale.xFor(p.year)}, ${scale.yFor(p.per1m) - 12})`} pointerEvents="none">
                        <text textAnchor="middle" style={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--fg)", paintOrder: "stroke", stroke: "var(--bg)", strokeWidth: 4 }}>
                          {p.year}: {p.per1m.toFixed(1)}/M ({p.hits.toLocaleString()}×)
                        </text>
                      </g>
                    )}
                  </g>
                );
              })}
            </svg>
          ) : null}
        </div>

        <div className="cx-coll-legend">
          <span className="cx-coll-leg-note">
            hits per 1M tokens, by document year
            {points && points.length > 0 ? ` · ${points.length} year${points.length === 1 ? "" : "s"}` : ""}
            {undated > 0 ? ` · ${undated.toLocaleString()} undated doc${undated === 1 ? "" : "s"} excluded` : ""}
          </span>
        </div>
      </div>
    </div>
  );
}
