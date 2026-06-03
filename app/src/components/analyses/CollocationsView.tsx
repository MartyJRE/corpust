// Collocation visualisation — three swappable views over the same data,
// chosen with the "view" selector so the modes can be compared fairly.
// All three share the same pan/zoom canvas (scroll = zoom, drag = pan,
// "fit" button = centre on content, slider = zoom level):
//
//   network   — GraphColl-style radial graph. Node at centre; collocates
//               radiate out, angle = left/right preference, distance =
//               association strength (closer = stronger). A collision
//               relaxation keeps the word labels from overlapping — the
//               thing that makes a dense scatter unreadable.
//   scatter   — x = left/right preference, y = score (axis auto-fit to the
//               data band). Greedy label declutter.
//   declutter — the same scatter, labelling only the strongest few +
//               whatever's hovered. Clean at a glance.
//
// All pure SVG + stdlib math; no d3 in the bundle.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { COLLOCATIONS } from "@/data";
import type { CollMetric, Collocate, CorpusMeta, QueryLayer } from "@/types";

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

const colorOf = (pos: string) => POS_FAMILY_COLOR[posFamily(pos)];

function prefRatio(c: Collocate): number {
  if (c.total === 0) return 0;
  return (c.rightCount - c.leftCount) / c.total;
}

const rFor = (total: number, maxTotal: number) =>
  4 + (Math.sqrt(total) / Math.sqrt(Math.max(1, maxTotal))) * 11;

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

// Word labels render in the sans face (Inter) — it has clean glyphs for
// the diacritics / ligatures (æ, œ, …) that show up in real corpus words
// and that a monospace face mangles. Because sans is proportional, label
// widths are measured exactly via a canvas so the collision layout stays
// tight instead of guessing from character count.
const LABEL_FONT = '12px "Inter", system-ui, sans-serif';
const NODE_FONT = '600 14px "Inter", system-ui, sans-serif';
let _measureCtx: CanvasRenderingContext2D | null = null;
function measureWidth(text: string, font: string): number {
  if (typeof document === "undefined") return text.length * 7;
  if (!_measureCtx) _measureCtx = document.createElement("canvas").getContext("2d");
  if (!_measureCtx) return text.length * 7;
  _measureCtx.font = font;
  return _measureCtx.measureText(text).width;
}

/** Fit a value axis to the data's actual range (plus padding) rather than
 *  anchoring at zero — logDice clusters in a narrow high band, so a
 *  0-based axis wastes most of the canvas. */
function fitDomain(values: number[]): [number, number] {
  if (values.length === 0) return [0, 1];
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

/** Evenly spaced "nice" ticks across a signed domain. */
function axisTicks(min: number, max: number, count = 5): number[] {
  const span = max - min || 1;
  const rawStep = span / (count - 1);
  const pow = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const step = Math.ceil(rawStep / pow) * pow || 1;
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 0.001; v += step) {
    ticks.push(Number(v.toFixed(6)));
  }
  return ticks;
}

const fmtTick = (t: number) => (Math.abs(t) >= 10 ? t.toFixed(0) : t.toFixed(1));

const W = 820;
const H = 460;
const M = { top: 28, right: 36, bottom: 52, left: 56 };
const PW = W - M.left - M.right;
const PH = H - M.top - M.bottom;
const ZOOM_MIN = 0.4;
const ZOOM_MAX = 16;

interface Bounds {
  x: number;
  y: number;
  w: number;
  h: number;
}

type VizMode = "network" | "scatter";

export interface CollocationsViewProps {
  corpus: CorpusMeta | null;
  term: string;
  /** Active query layer — lets the empty state explain that the POS layer
   *  expects a tag (NN, VB, …) rather than a word. */
  layer?: QueryLayer;
  /** Real collocates from the backend. When null, falls back to the
   *  fixture `COLLOCATIONS` so the view still looks populated in
   *  demos / non-Tauri preview. */
  data?: Collocate[] | null;
  /** Backend scan in flight — show a loading overlay. */
  loading?: boolean;
  /** The scan hit its safety ceiling; scores are from a large sample. */
  truncated?: boolean;
  /** Tokens on the left of the node to consider (0 = skip). */
  leftWindow?: number;
  rightWindow?: number;
  onWindowChange?: (left: number, right: number) => void;
}

const WINDOW_CHOICES = [0, 1, 2, 3, 5, 7, 10] as const;

type CollSortKey = "word" | "pos" | "leftCount" | "rightCount" | "total" | "logDice" | "mi" | "z";

export function CollocationsView({
  term,
  layer,
  data: dataProp,
  loading = false,
  truncated = false,
  leftWindow = 5,
  rightWindow = 5,
  onWindowChange,
}: CollocationsViewProps) {
  const [metric, setMetric] = useState<CollMetric>("logDice");
  const [vizMode, setVizMode] = useState<VizMode>("network");
  const [declutter, setDeclutter] = useState(false);
  const [hover, setHover] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<CollSortKey>("total");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const toggleSort = (key: CollSortKey, defaultDir: "asc" | "desc" = "desc") => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(defaultDir);
    }
  };

  // `dataProp` is null only for fixture/demo corpora (App never queried the
  // backend); a live corpus always passes an array — possibly empty. Don't
  // fall back to the demo fixture for a live empty result (that showed the
  // coloured demo collocates and looked like a real, colourful answer).
  const isLiveData = dataProp != null;
  const rawData = useMemo(() => (isLiveData ? dataProp : COLLOCATIONS), [dataProp, isLiveData]);
  const isEmptyLive = isLiveData && rawData.length === 0;

  // The chart uses a stable order (by total, frequency) independent of the
  // table's sort — sorting the table below must not reshuffle the graph.
  const chartData = useMemo(() => [...rawData].sort((a, b) => b.total - a.total), [rawData]);
  // The table gets its own sort.
  const tableData = useMemo(() => {
    const copy = [...rawData];
    copy.sort((a, b) => {
      const av = a[sortKey] as unknown as number | string;
      const bv = b[sortKey] as unknown as number | string;
      if (typeof av === "number" && typeof bv === "number") return av - bv;
      return String(av).localeCompare(String(bv));
    });
    if (sortDir === "desc") copy.reverse();
    return copy;
  }, [rawData, sortKey, sortDir]);

  const [domMin, domMax] = useMemo(() => fitDomain(rawData.map((d) => d[metric])), [rawData, metric]);
  const maxTotal = useMemo(() => Math.max(1, ...rawData.map((d) => d.total)), [rawData]);
  const meterWidth = (score: number) =>
    Math.max(4, ((score - domMin) / (domMax - domMin || 1)) * 120);

  const metricLabel = metric === "logDice" ? "log-Dice" : metric === "mi" ? "MI" : "z-score";

  const chartProps: ChartProps = {
    data: chartData,
    metric,
    term,
    domMin,
    domMax,
    maxTotal,
    hover,
    setHover,
    metricLabel,
  };

  return (
    <div className="cx-coll-wrap">
      <div className="cx-coll-main">
        <div className="cx-coll-head">
          <h2 className="cx-coll-title">
            collocates of <span className="kw">{term}</span>
            {truncated && (
              <span
                title="The node occurs more often than the scan ceiling; scores are from a large sample, not every occurrence."
                style={{
                  marginLeft: 8,
                  fontFamily: "var(--font-mono)",
                  fontSize: 10,
                  fontWeight: 400,
                  color: "var(--warn)",
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                }}
              >
                · sampled
              </span>
            )}
          </h2>
          <div className="cx-coll-controls">
            <span>view</span>
            <div className="cx-coll-segbtn">
              {(
                [
                  ["network", "network"],
                  ["scatter", "scatter"],
                ] as const
              ).map(([k, l]) => (
                <button
                  key={k}
                  type="button"
                  className={vizMode === k ? "is-on" : ""}
                  onClick={() => setVizMode(k)}
                >
                  {l}
                </button>
              ))}
            </div>
            {vizMode === "scatter" && (
              <div className="cx-coll-segbtn">
                <button
                  type="button"
                  className={declutter ? "is-on" : ""}
                  onClick={() => setDeclutter((d) => !d)}
                  title="thin to the strongest labels and hide clashes"
                >
                  declutter
                </button>
              </div>
            )}
            <span style={{ marginLeft: 12 }}>L</span>
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

        <div className="cx-coll-graph cx-coll-svg-wrap" style={{ position: "relative" }}>
          {loading ? (
            // While a query runs, render *only* the overlay — never the
            // chart with stale data, which would mount with the wrong
            // bounds and then re-fit (the "huge centre node, then jump").
            <div
              style={{
                position: "absolute",
                inset: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                color: "var(--fg-muted)",
                letterSpacing: "0.04em",
              }}
            >
              computing collocations…
            </div>
          ) : isEmptyLive ? (
            <div
              style={{
                position: "absolute",
                inset: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                color: "var(--fg-subtle)",
                letterSpacing: "0.03em",
                textAlign: "center",
                padding: 24,
              }}
            >
              {!term.trim()
                ? "enter a search term to see its collocates"
                : layer === "pos"
                  ? `no matches for “${term}” on the POS layer — search a tag (NN, VB, JJ…)`
                  : `no collocates for “${term}” on this layer`}
            </div>
          ) : vizMode === "network" ? (
            <NetworkChart {...chartProps} />
          ) : (
            <ScatterChart {...chartProps} labelMode={declutter ? "topN" : "all"} />
          )}
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
          <span className="cx-coll-leg-note">
            scroll = zoom · drag = pan · ⤢ = centre · dot size ∝ frequency
          </span>
        </div>

        <table className="cx-coll-table">
          <thead>
            <tr>
              <SortTh label="collocate" skey="word" align="left" sortKey={sortKey} sortDir={sortDir} onClick={() => toggleSort("word", "asc")} />
              <SortTh label="pos" skey="pos" align="left" sortKey={sortKey} sortDir={sortDir} onClick={() => toggleSort("pos", "asc")} />
              <SortTh label="L" skey="leftCount" align="right" sortKey={sortKey} sortDir={sortDir} onClick={() => toggleSort("leftCount")} />
              <SortTh label="R" skey="rightCount" align="right" sortKey={sortKey} sortDir={sortDir} onClick={() => toggleSort("rightCount")} />
              <SortTh label="total" skey="total" align="right" sortKey={sortKey} sortDir={sortDir} onClick={() => toggleSort("total")} />
              <SortTh label="logDice" skey="logDice" align="right" sortKey={sortKey} sortDir={sortDir} onClick={() => toggleSort("logDice")} />
              <SortTh label="MI" skey="mi" align="right" sortKey={sortKey} sortDir={sortDir} onClick={() => toggleSort("mi")} />
              <SortTh label="z-score" skey="z" align="right" sortKey={sortKey} sortDir={sortDir} onClick={() => toggleSort("z")} />
              <th className="num">strength</th>
            </tr>
          </thead>
          <tbody>
            {tableData.map((c) => (
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
                  <span className="cx-meter" style={{ width: meterWidth(c[metric]) }} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface ChartProps {
  data: Collocate[];
  metric: CollMetric;
  term: string;
  domMin: number;
  domMax: number;
  maxTotal: number;
  hover: string | null;
  setHover: (w: string | null) => void;
  metricLabel: string;
}

// ---------------------------------------------------------------------------
// Shared pan/zoom canvas
// ---------------------------------------------------------------------------

/** Transform that fits content `bounds` (base coords) into the viewport. */
function fitTransform(b: Bounds): { k: number; tx: number; ty: number } {
  const k = clamp(Math.min(W / b.w, H / b.h) * 0.92, ZOOM_MIN, ZOOM_MAX);
  return { k, tx: W / 2 - k * (b.x + b.w / 2), ty: H / 2 - k * (b.y + b.h / 2) };
}

/** SVG with scroll-to-zoom, drag-to-pan, a "fit" button and a zoom
 *  slider. Children are drawn in base [0..W]×[0..H] coords; the transform
 *  scales the whole picture uniformly (so labels/dots/grid stay aligned).
 *  `bounds` is the content extent in base coords, used by the fit control;
 *  `autoFit` fits once when the bounds change (used by the network layout,
 *  whose extent varies). */
function ZoomPanSvg({
  bounds,
  autoFit,
  children,
  onLeave,
}: {
  bounds: Bounds;
  autoFit: boolean;
  children: React.ReactNode;
  /** Called when the pointer leaves the chart — used to clear any stuck
   *  hover (a node raised-to-front can otherwise miss its mouseleave). */
  onLeave?: () => void;
}) {
  // Initialise already-fitted (when autoFit) so the first paint is the
  // centred view — no identity frame where the centre node renders huge
  // before the fit effect kicks in.
  const [t, setT] = useState(() => (autoFit ? fitTransform(bounds) : { k: 1, tx: 0, ty: 0 }));
  const tRef = useRef(t);
  tRef.current = t;
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const drag = useRef<{ cx: number; cy: number; tx: number; ty: number } | null>(null);
  const [dragging, setDragging] = useState(false);

  const fitTo = useCallback((b: Bounds) => setT(fitTransform(b)), []);

  // Re-fit when the content extent changes (e.g. a new query while the
  // network view is already open). The initial fit is handled by the
  // lazy state above, so this only matters for subsequent bounds changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (autoFit) fitTo(bounds);
  }, [autoFit, bounds.x, bounds.y, bounds.w, bounds.h]);

  // Non-passive wheel listener on the wrapper so zooming never scrolls the
  // page, even over the chart's letterboxed margins.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const sx = ((e.clientX - rect.left) / rect.width) * W;
      const sy = ((e.clientY - rect.top) / rect.height) * H;
      const cur = tRef.current;
      const k = clamp(cur.k * Math.exp(-e.deltaY * 0.0012), ZOOM_MIN, ZOOM_MAX);
      const ratio = k / cur.k;
      setT({ k, tx: sx - (sx - cur.tx) * ratio, ty: sy - (sy - cur.ty) * ratio });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const onMouseDown = (e: React.MouseEvent) => {
    e.preventDefault(); // don't start a text selection while panning
    drag.current = { cx: e.clientX, cy: e.clientY, tx: t.tx, ty: t.ty };
    setDragging(true);
    onLeave?.(); // dropping into a pan shouldn't leave a node stuck-hovered
  };
  const onMouseMove = (e: React.MouseEvent) => {
    const d = drag.current;
    const el = wrapRef.current;
    if (!d || !el) return;
    const rect = el.getBoundingClientRect();
    const dx = ((e.clientX - d.cx) / rect.width) * W;
    const dy = ((e.clientY - d.cy) / rect.height) * H;
    setT((cur) => ({ ...cur, tx: d.tx + dx, ty: d.ty + dy }));
  };
  const endDrag = () => {
    drag.current = null;
    setDragging(false);
  };

  const setZoom = (level: number) => {
    setT((cur) => {
      const ratio = level / cur.k;
      return { k: level, tx: W / 2 - (W / 2 - cur.tx) * ratio, ty: H / 2 - (H / 2 - cur.ty) * ratio };
    });
  };

  return (
    <div ref={wrapRef} style={{ position: "absolute", inset: 0, userSelect: "none", WebkitUserSelect: "none" }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        style={{ width: "100%", height: "100%", display: "block", cursor: dragging ? "grabbing" : "grab", userSelect: "none", WebkitUserSelect: "none" }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={endDrag}
        onMouseLeave={() => {
          endDrag();
          onLeave?.();
        }}
        onDoubleClick={() => fitTo(bounds)}
      >
        <g transform={`translate(${t.tx} ${t.ty}) scale(${t.k})`}>{children}</g>
      </svg>

      {/* controls */}
      <div
        style={{
          position: "absolute",
          top: 10,
          right: 10,
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "4px 8px",
          background: "color-mix(in oklch, var(--bg-raised) 88%, transparent)",
          border: "1px solid var(--border)",
          borderRadius: 6,
        }}
      >
        <button
          type="button"
          onClick={() => fitTo(bounds)}
          title="centre on content"
          style={{
            border: 0,
            background: "transparent",
            color: "var(--fg-muted)",
            cursor: "pointer",
            fontSize: 14,
            lineHeight: 1,
            padding: 0,
          }}
        >
          ⤢
        </button>
        <input
          type="range"
          min={ZOOM_MIN}
          max={ZOOM_MAX}
          step={0.01}
          value={t.k}
          onChange={(e) => setZoom(Number(e.target.value))}
          title={`zoom ${t.k.toFixed(1)}×`}
          style={{ width: 96, accentColor: "var(--accent)" }}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Network (GraphColl-style radial graph)
// ---------------------------------------------------------------------------

function NetworkChart({ data, metric, term, domMin, domMax, maxTotal, hover, setHover, metricLabel }: ChartProps) {
  const { cx0, cy0, nodes, bounds, nodeR } = useMemo(() => {
    const cx0 = W / 2;
    const cy0 = H / 2;
    // Centre node is a circle big enough to hold the full term.
    const nodeR = clamp(measureWidth(term, NODE_FONT) / 2 + 16, 30, 150);
    const minR = Math.max(86, nodeR + 34);
    const maxR = Math.min(W, H) / 2 - 26;
    const span = domMax - domMin || 1;

    // Even radial spread: distance encodes strength (strong → near the
    // node), angle is an even golden-angle distribution around the full
    // circle. The collision pass below settles overlaps.
    const GOLDEN = Math.PI * (3 - Math.sqrt(5));
    const nodes = data.map((c, i) => {
      const t = clamp((c[metric] - domMin) / span, 0, 1); // 1 = strongest
      const radius = maxR - t * (maxR - minR); // strong → close to node
      const angle = i * GOLDEN;
      return {
        c,
        x: cx0 + radius * Math.cos(angle),
        y: cy0 - radius * Math.sin(angle),
        w: measureWidth(c.word, LABEL_FONT) + 16,
        h: 16,
      };
    });

    // Collision relaxation: push overlapping label boxes apart, keep them
    // clear of the node chip and inside the canvas. Deterministic.
    for (let iter = 0; iter < 160; iter++) {
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i];
          const b = nodes[j];
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const ox = (a.w + b.w) / 2 - Math.abs(dx);
          const oy = (a.h + b.h) / 2 + 3 - Math.abs(dy);
          if (ox > 0 && oy > 0) {
            if (ox < oy) {
              const m = (ox / 2) * (dx >= 0 ? 1 : -1);
              a.x -= m;
              b.x += m;
            } else {
              const m = (oy / 2) * (dy >= 0 ? 1 : -1);
              a.y -= m;
              b.y += m;
            }
          }
        }
      }
      for (const n of nodes) {
        const dx = n.x - cx0;
        const dy = n.y - cy0;
        const dist = Math.hypot(dx, dy) || 1;
        if (dist < minR) {
          n.x = cx0 + (dx / dist) * minR;
          n.y = cy0 + (dy / dist) * minR;
        }
      }
    }

    // Content extent (include label widths) for the fit control.
    let x0 = cx0;
    let y0 = cy0;
    let x1 = cx0;
    let y1 = cy0;
    for (const n of nodes) {
      x0 = Math.min(x0, n.x - n.w / 2);
      x1 = Math.max(x1, n.x + n.w / 2);
      y0 = Math.min(y0, n.y - 10);
      y1 = Math.max(y1, n.y + 10);
    }
    const pad = 20;
    const bounds: Bounds = { x: x0 - pad, y: y0 - pad, w: x1 - x0 + 2 * pad, h: y1 - y0 + 2 * pad };
    return { cx0, cy0, nodes, bounds, nodeR };
  }, [data, metric, domMin, domMax, term]);

  // Draw the hovered node last so it sits above its neighbours (the
  // labels are scattered, not stacked, so the active one must come to
  // front to read clearly).
  const ordered = hover ? [...nodes].sort((a, b) => (a.c.word === hover ? 1 : b.c.word === hover ? -1 : 0)) : nodes;

  return (
    <ZoomPanSvg bounds={bounds} autoFit onLeave={() => setHover(null)}>
      {/* edges */}
      {nodes.map((n) => {
        const isHover = hover === n.c.word;
        const dimmed = hover !== null && !isHover;
        return (
          <line
            key={`e-${n.c.word}`}
            x1={cx0}
            y1={cy0}
            x2={n.x}
            y2={n.y}
            stroke={isHover ? colorOf(n.c.pos) : "var(--border)"}
            strokeWidth={isHover ? 1.5 : 1}
            opacity={dimmed ? 0.12 : isHover ? 0.7 : 0.3}
          />
        );
      })}

      {/* node centre — a circle big enough for the full term */}
      <circle cx={cx0} cy={cy0} r={nodeR} fill="var(--bg)" stroke="var(--accent)" strokeWidth={1.5} />
      <text x={cx0} y={cy0 + 5} textAnchor="middle" style={{ fontFamily: "var(--font-sans)", fontSize: 14, fontWeight: 600, fill: "var(--accent)" }}>
        {term}
      </text>

      {/* collocates */}
      {ordered.map((n) => {
        const c = n.c;
        const isHover = hover === c.word;
        const dimmed = hover !== null && !isHover;
        const color = colorOf(c.pos);
        const labelW = measureWidth(c.word, LABEL_FONT);
        const dot = rFor(c.total, maxTotal);
        const half = Math.max(labelW / 2, dot);
        return (
          <g
            key={c.word}
            transform={`translate(${n.x}, ${n.y})`}
            style={{ cursor: "pointer", opacity: dimmed ? 0.28 : 1 }}
            onMouseEnter={() => setHover(c.word)}
            onMouseLeave={() => setHover(null)}
          >
            {/* transparent hit area covering dot + label so hovering the
                text (not just the dot) selects the node */}
            <rect x={-half - 4} y={-12} width={2 * half + 8} height={24} fill="transparent" />
            <circle
              r={dot}
              fill={color}
              stroke={color}
              pointerEvents="none"
              style={{
                fillOpacity: isHover ? 0.5 : 0.18,
                strokeWidth: isHover ? 2 : 1,
                transition: "fill-opacity 130ms var(--ease), stroke-width 130ms var(--ease)",
              }}
            />
            <text
              y={4}
              textAnchor="middle"
              style={{
                fontFamily: "var(--font-sans)",
                fontSize: 12,
                fontWeight: 500,
                fill: isHover ? "var(--accent)" : "var(--fg)",
                pointerEvents: "none",
                paintOrder: "stroke",
                stroke: "var(--bg)",
                strokeWidth: 4,
                transition: "fill 130ms var(--ease)",
              }}
            >
              {c.word}
            </text>
            {isHover &&
              (() => {
                const tip = `${metricLabel} ${c[metric].toFixed(1)} · ${c.total.toLocaleString()}×`;
                const tw = measureWidth(tip, '10px "JetBrains Mono", monospace');
                return (
                  <g pointerEvents="none">
                    {/* opaque backing so the readout stays legible over nodes behind it */}
                    <rect x={-tw / 2 - 6} y={13} width={tw + 12} height={17} rx={4} fill="var(--bg)" fillOpacity={0.92} stroke="var(--border)" strokeWidth={1} />
                    <text y={25} textAnchor="middle" style={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--fg-muted)" }}>
                      {tip}
                    </text>
                  </g>
                );
              })()}
          </g>
        );
      })}
    </ZoomPanSvg>
  );
}

// ---------------------------------------------------------------------------
// Scatter (x = L/R preference, y = score) + declutter
// ---------------------------------------------------------------------------

function ScatterChart({
  data,
  metric,
  term,
  domMin,
  domMax,
  maxTotal,
  hover,
  setHover,
  metricLabel,
  labelMode,
}: ChartProps & { labelMode: "all" | "topN" }) {
  const X0 = -1.05;
  const X1 = 1.05;
  const xBase = (pref: number) => M.left + ((pref - X0) / (X1 - X0)) * PW;
  const yBase = (score: number) => M.top + PH - ((score - domMin) / (domMax - domMin || 1)) * PH;

  const yTicks = axisTicks(domMin, domMax, 5);
  const xTicks = [-1, -0.5, 0, 0.5, 1];

  // Which labels render. "all" (scatter) shows every label — overlaps and
  // all, that's what pan/zoom is for. "topN" (declutter) keeps only the
  // strongest few and drops any that would clash, for a clean read.
  const labelled = useMemo(() => {
    if (labelMode === "all") return new Set(data.map((c) => c.word));
    const byScore = [...data].sort((a, b) => b[metric] - a[metric]);
    const pool = byScore.slice(0, 14);
    const placed: Bounds[] = [];
    const show = new Set<string>();
    for (const c of pool) {
      const cx = xBase(prefRatio(c)) + rFor(c.total, maxTotal) + 5;
      const cy = yBase(c[metric]);
      const w = measureWidth(c.word, LABEL_FONT);
      const box: Bounds = { x: cx, y: cy - 7, w, h: 14 };
      const clash = placed.some(
        (p) => box.x < p.x + p.w && box.x + box.w > p.x && box.y < p.y + p.h && box.y + box.h > p.y,
      );
      if (!clash) {
        placed.push(box);
        show.add(c.word);
      }
    }
    return show;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, metric, domMin, domMax, labelMode, maxTotal]);

  // "all" mode: keep dots at their true position but push the *labels*
  // apart (collision relaxation, anchored near each dot) so every label is
  // readable; a leader line links each label back to its dot. Returns a
  // map word → displaced label centre. null in declutter mode.
  const placed = useMemo(() => {
    if (labelMode !== "all") return null;
    const arr = data.map((c) => {
      const cx = xBase(prefRatio(c));
      const cy = yBase(c[metric]);
      const r = rFor(c.total, maxTotal);
      const w = measureWidth(c.word, LABEL_FONT);
      const ax = cx + r + 7 + w / 2; // anchor: just right of the dot
      return { word: c.word, w, x: ax, y: cy, ax, ay: cy };
    });
    for (let it = 0; it < 260; it++) {
      for (let i = 0; i < arr.length; i++) {
        for (let j = i + 1; j < arr.length; j++) {
          const a = arr[i];
          const b = arr[j];
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const ox = (a.w + b.w) / 2 + 8 - Math.abs(dx);
          const oy = 19 - Math.abs(dy);
          if (ox > 0 && oy > 0) {
            if (ox < oy) {
              const m = (ox / 2) * (dx >= 0 ? 1 : -1);
              a.x -= m;
              b.x += m;
            } else {
              const m = (oy / 2) * (dy >= 0 ? 1 : -1);
              a.y -= m;
              b.y += m;
            }
          }
        }
      }
      for (const l of arr) {
        // Weak pull back toward the dot — loose enough that a dense pile-up
        // can fan out into empty space rather than staying stacked.
        l.x += (l.ax - l.x) * 0.02;
        l.y += (l.ay - l.y) * 0.02;
        l.x = clamp(l.x, M.left + l.w / 2, W - M.right - l.w / 2);
        l.y = clamp(l.y, M.top + 8, M.top + PH - 8);
      }
    }
    const m = new Map<string, { x: number; y: number; w: number }>();
    for (const l of arr) m.set(l.word, { x: l.x, y: l.y, w: l.w });
    return m;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, metric, domMin, domMax, labelMode, maxTotal]);

  // Data extent for the fit control.
  const bounds = useMemo(() => {
    if (data.length === 0) return { x: M.left, y: M.top, w: PW, h: PH };
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    for (const c of data) {
      const px = xBase(prefRatio(c));
      const py = yBase(c[metric]);
      x0 = Math.min(x0, px);
      y0 = Math.min(y0, py);
      x1 = Math.max(x1, px);
      y1 = Math.max(y1, py);
    }
    const pad = 36;
    return { x: x0 - pad, y: y0 - pad, w: x1 - x0 + 2 * pad, h: y1 - y0 + 2 * pad };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, metric, domMin, domMax]);

  const nodeX = xBase(0);

  return (
    <ZoomPanSvg bounds={bounds} autoFit={false} onLeave={() => setHover(null)}>
      {/* Y gridlines + labels */}
      {yTicks.map((t) => (
        <g key={`y-${t}`}>
          <line x1={M.left} x2={W - M.right} y1={yBase(t)} y2={yBase(t)} stroke="var(--border)" strokeWidth={1} opacity={0.32} />
          <text x={M.left - 10} y={yBase(t)} textAnchor="end" dominantBaseline="middle" style={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--fg-subtle)" }}>
            {fmtTick(t)}
          </text>
        </g>
      ))}

      {/* Y axis title */}
      <text x={16} y={M.top + PH / 2} textAnchor="middle" dominantBaseline="middle" transform={`rotate(-90 16 ${M.top + PH / 2})`} style={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--fg-subtle)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
        {metricLabel}
      </text>

      {/* node (x=0) line */}
      <line x1={nodeX} x2={nodeX} y1={M.top - 4} y2={H - M.bottom + 4} stroke="var(--accent)" strokeWidth={1} strokeDasharray="3 4" opacity={0.4} />
      <text x={nodeX} y={M.top - 10} textAnchor="middle" style={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--accent)", fontWeight: 600 }}>
        {term}
      </text>

      {/* X baseline + ticks */}
      <line x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} stroke="var(--border)" />
      {xTicks.map((t) => (
        <g key={`x-${t}`}>
          <line x1={xBase(t)} x2={xBase(t)} y1={H - M.bottom} y2={H - M.bottom + 4} stroke="var(--border)" />
          <text x={xBase(t)} y={H - M.bottom + 18} textAnchor="middle" style={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--fg-subtle)" }}>
            {t === 0 ? "" : t > 0 ? `+${t}` : `${t}`}
          </text>
        </g>
      ))}
      <text x={M.left} y={H - 8} textAnchor="start" style={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--fg-subtle)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
        ← left of node
      </text>
      <text x={W - M.right} y={H - 8} textAnchor="end" style={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--fg-subtle)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
        right of node →
      </text>

      {/* Data points — hovered drawn last so it reads above its neighbours */}
      {(hover ? [...data].sort((a, b) => (a.word === hover ? 1 : b.word === hover ? -1 : 0)) : data).map((c) => {
        const cx = xBase(prefRatio(c));
        const cy = yBase(c[metric]);
        const r = rFor(c.total, maxTotal);
        const color = colorOf(c.pos);
        const isHover = hover === c.word;
        const dimmed = hover !== null && !isHover;

        // "all" mode: dot at true position, label displaced + leader line.
        if (placed) {
          const p = placed.get(c.word);
          if (!p) return null;
          return (
            <g
              key={c.word}
              style={{ cursor: "pointer", opacity: dimmed ? 0.28 : 1 }}
              onMouseEnter={() => setHover(c.word)}
              onMouseLeave={() => setHover(null)}
            >
              <line x1={cx + r} y1={cy} x2={p.x - p.w / 2 - 3} y2={p.y} stroke={isHover ? color : "var(--border)"} strokeWidth={1} opacity={isHover ? 0.7 : 0.35} />
              {/* hit area is the label only (tight to the glyphs) — every
                  label is visible in this mode, so the dot needn't be a
                  separate, far-from-the-word hit target */}
              <rect x={p.x - p.w / 2} y={p.y - 8} width={p.w} height={15} fill="transparent" />
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill={color}
                stroke={color}
                pointerEvents="none"
                style={{
                  fillOpacity: isHover ? 0.5 : 0.22,
                  strokeWidth: isHover ? 2 : 1.2,
                  transition: "fill-opacity 130ms var(--ease), stroke-width 130ms var(--ease)",
                }}
              />
              <text
                x={p.x}
                y={p.y + 4}
                textAnchor="middle"
                style={{
                  fontFamily: "var(--font-sans)",
                  fontSize: 12,
                  fontWeight: 500,
                  fill: isHover ? "var(--accent)" : "var(--fg)",
                  pointerEvents: "none",
                  paintOrder: "stroke",
                  stroke: "var(--bg)",
                  strokeWidth: 4,
                  transition: "fill 130ms var(--ease)",
                }}
              >
                {c.word}
              </text>
            </g>
          );
        }

        // declutter mode: glued labels, only the top-N shown.
        const hasLabel = labelled.has(c.word);
        const showLabel = isHover || hasLabel;
        const labelW = measureWidth(c.word, LABEL_FONT);
        const hy = Math.max(r + 2, 11);
        const hitW = showLabel ? r + 9 + labelW : 2 * r + 8;
        const hitX = showLabel ? -r - 2 : -r - 4;
        return (
          <g
            key={c.word}
            transform={`translate(${cx}, ${cy})`}
            style={{ cursor: "pointer", opacity: dimmed ? 0.28 : 1 }}
            onMouseEnter={() => setHover(c.word)}
            onMouseLeave={() => setHover(null)}
          >
            <rect x={hitX} y={-hy} width={hitW} height={2 * hy} fill="transparent" />
            <circle
              r={r}
              fill={color}
              stroke={color}
              pointerEvents="none"
              style={{
                fillOpacity: isHover ? 0.5 : 0.22,
                strokeWidth: isHover ? 2 : 1.2,
                transition: "fill-opacity 130ms var(--ease), stroke-width 130ms var(--ease)",
              }}
            />
            {showLabel && (
              <text
                x={r + 5}
                y={4}
                style={{
                  fontFamily: "var(--font-sans)",
                  fontSize: 12,
                  fontWeight: 500,
                  fill: isHover ? "var(--accent)" : "var(--fg)",
                  pointerEvents: "none",
                  paintOrder: "stroke",
                  stroke: "var(--bg)",
                  strokeWidth: 4,
                  transition: "fill 130ms var(--ease)",
                }}
              >
                {c.word}
              </text>
            )}
          </g>
        );
      })}
    </ZoomPanSvg>
  );
}

interface SortThProps {
  label: string;
  skey: CollSortKey;
  align: "left" | "right";
  sortKey: CollSortKey;
  sortDir: "asc" | "desc";
  onClick: () => void;
}

function SortTh({ label, skey, align, sortKey, sortDir, onClick }: SortThProps) {
  const active = sortKey === skey;
  const arrow = active ? (sortDir === "asc" ? "↑" : "↓") : "";
  return (
    <th
      className={align === "right" ? "num" : undefined}
      onClick={onClick}
      style={{ cursor: "pointer", userSelect: "none" }}
      title={active ? `click to toggle ${sortDir === "asc" ? "asc→desc" : "desc→asc"}` : `sort by ${label}`}
    >
      {label}
      <span style={{ opacity: active ? 1 : 0.3, marginLeft: 4 }}>{arrow || "↕"}</span>
    </th>
  );
}
