// Word tree — a branching view of what follows (or precedes) the node,
// à la Wattenberg & Viégas. Built client-side from the concordance hits:
// each hit's right (or left) context is a token sequence; we trie them up
// and lay the trie out left→right, sizing each word by how many lines run
// through it. Great for seeing phraseology at a glance — e.g. for a
// sanctions corpus, `gold →` reserves / bars / "of Russian origin".

import { useMemo, useState } from "react";
import type { KwicResult } from "@/types";

export interface WordTreeProps {
  corpusName: string;
  term: string;
  /** Concordance result; its hits supply the context sequences. */
  result: KwicResult | null;
  loading?: boolean;
}

const DEPTH = 4; // context tokens per branch
const MAX_CHILDREN = 6; // keep each node's busiest continuations
const COL_W = 148;
const ROW_H = 22;
const PAD_X = 16;

const CHAR_W = 7.4; // sans @ ~13px, rough — only for edge endpoints

interface Trie {
  word: string;
  count: number;
  children: Map<string, Trie>;
}

function tokenize(s: string): string[] {
  return s
    .split(/\s+/)
    .map((t) => t.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "").toLowerCase())
    .filter(Boolean);
}

interface Placed {
  id: number;
  word: string;
  count: number;
  depth: number;
  x: number;
  y: number;
  parent: number | null;
}

export function WordTree({ corpusName, term, result, loading = false }: WordTreeProps) {
  const [dir, setDir] = useState<"right" | "left">("right");

  const hits = result?.hits ?? [];

  const { placed, height, leaves } = useMemo(() => {
    // Build the trie from each hit's context, trimmed to DEPTH tokens.
    const root: Trie = { word: term, count: 0, children: new Map() };
    for (const h of hits) {
      let seq = tokenize(dir === "right" ? h.right : h.left);
      if (dir === "left") seq = seq.reverse(); // nearest-to-node first
      seq = seq.slice(0, DEPTH);
      root.count += 1;
      let node = root;
      for (const w of seq) {
        let child = node.children.get(w);
        if (!child) {
          child = { word: w, count: 0, children: new Map() };
          node.children.set(w, child);
        }
        child.count += 1;
        node = child;
      }
    }

    // Count leaves (for canvas height) after pruning to MAX_CHILDREN.
    const prune = (n: Trie): Trie => ({
      ...n,
      children: new Map(
        [...n.children.values()]
          .sort((a, b) => b.count - a.count)
          .slice(0, MAX_CHILDREN)
          .map((c) => [c.word, prune(c)]),
      ),
    });
    const pruned = prune(root);

    const countLeaves = (n: Trie): number => {
      if (n.children.size === 0) return 1;
      return [...n.children.values()].reduce((s, c) => s + countLeaves(c), 0);
    };
    const leaves = countLeaves(pruned);
    const height = Math.max(360, leaves * ROW_H + 40);

    // Band layout: each node owns a vertical band sized by its frequency;
    // it sits at the band's centre, children split the band beneath it.
    const placed: Placed[] = [];
    let nextId = 0;
    const walk = (n: Trie, depth: number, yTop: number, yBottom: number, parent: number | null) => {
      const id = nextId++;
      const y = (yTop + yBottom) / 2;
      const x = dir === "right" ? PAD_X + depth * COL_W : 0; // x for left dir fixed up after width known
      placed.push({ id, word: n.word, count: n.count, depth, x, y, parent });
      const kids = [...n.children.values()];
      const total = kids.reduce((s, k) => s + k.count, 0) || 1;
      let cursor = yTop;
      for (const k of kids) {
        const band = (k.count / total) * (yBottom - yTop);
        walk(k, depth + 1, cursor, cursor + band, id);
        cursor += band;
      }
    };
    walk(pruned, 0, 20, height - 20, null);

    return { placed, height, leaves };
  }, [hits, term, dir]);

  const maxDepth = placed.reduce((m, p) => Math.max(m, p.depth), 0);
  const width = PAD_X * 2 + (maxDepth + 1) * COL_W;

  // For the left direction we mirror x so the node sits on the right and
  // branches grow leftward.
  const xOf = (p: Placed) => (dir === "right" ? p.x : width - PAD_X - p.depth * COL_W);
  const byId = new Map(placed.map((p) => [p.id, p]));

  const maxCount = placed.reduce((m, p) => Math.max(m, p.count), 1);
  const fontFor = (count: number) => Math.min(28, 12 + Math.log2(count + 1) * 1.8);

  // Hovering a word lights up the branch it belongs to: its ancestors
  // (the phrase leading to it) and its descendants (the continuations).
  const [hoverId, setHoverId] = useState<number | null>(null);
  const childrenOf = useMemo(() => {
    const m = new Map<number, number[]>();
    for (const p of placed) {
      if (p.parent != null) {
        const arr = m.get(p.parent) ?? [];
        arr.push(p.id);
        m.set(p.parent, arr);
      }
    }
    return m;
  }, [placed]);
  const connected = useMemo(() => {
    if (hoverId == null) return null;
    const set = new Set<number>();
    // ancestors (incl. self)
    let cur: number | null = hoverId;
    while (cur != null) {
      set.add(cur);
      cur = byId.get(cur)?.parent ?? null;
    }
    // descendants
    const stack = [hoverId];
    while (stack.length) {
      const id = stack.pop() as number;
      for (const c of childrenOf.get(id) ?? []) {
        set.add(c);
        stack.push(c);
      }
    }
    return set;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hoverId, placed]);
  const on = (id: number) => connected == null || connected.has(id);

  const empty = !loading && hits.length === 0;

  return (
    <div className="cx-coll-wrap">
      <div className="cx-coll-main">
        <div className="cx-coll-head">
          <h2 className="cx-coll-title">
            word tree · <span className="kw">{term}</span>{" "}
            <span style={{ color: "var(--fg-muted)", fontSize: 14 }}>· {corpusName}</span>
          </h2>
          <div className="cx-coll-controls">
            <span>direction</span>
            <div className="cx-coll-segbtn">
              {(
                [
                  ["right", "following →"],
                  ["left", "← preceding"],
                ] as const
              ).map(([k, l]) => (
                <button key={k} type="button" className={dir === k ? "is-on" : ""} onClick={() => setDir(k)}>
                  {l}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="cx-coll-graph" style={{ overflow: "auto", height: "clamp(460px, 60vh, 780px)" }}>
          {loading ? (
            <div className="cx-loading-row" style={{ justifyContent: "center", height: "100%" }}>
              <span className="cx-spinner" /> building tree…
            </div>
          ) : empty ? (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                color: "var(--fg-subtle)",
              }}
            >
              no concordance lines for “{term}” to branch from
            </div>
          ) : (
            <svg width={width} height={height} style={{ display: "block", fontFamily: "var(--font-sans)" }}>
              {/* edges — an edge lights up when both ends are on the
                  hovered branch */}
              {placed.map((p) => {
                if (p.parent == null) return null;
                const parent = byId.get(p.parent);
                if (!parent) return null;
                const px = dir === "right" ? xOf(parent) + parent.word.length * CHAR_W + 6 : xOf(parent) - parent.word.length * CHAR_W - 6;
                const cx = dir === "right" ? xOf(p) - 6 : xOf(p) + 6;
                const mx = (px + cx) / 2;
                const lit = on(p.id) && on(p.parent);
                return (
                  <path
                    key={`e-${p.id}`}
                    d={`M ${px} ${parent.y} C ${mx} ${parent.y}, ${mx} ${p.y}, ${cx} ${p.y}`}
                    fill="none"
                    stroke={lit && connected != null ? "var(--accent)" : "var(--border)"}
                    strokeWidth={Math.max(0.6, Math.min(4, (p.count / maxCount) * 4))}
                    opacity={connected == null ? 0.5 : lit ? 0.9 : 0.08}
                    style={{ transition: "opacity 120ms var(--ease), stroke 120ms var(--ease)" }}
                  />
                );
              })}
              {/* nodes */}
              {placed.map((p) => {
                const isRoot = p.parent == null;
                const fs = isRoot ? 18 : fontFor(p.count);
                const lit = on(p.id);
                return (
                  <g
                    key={p.id}
                    transform={`translate(${xOf(p)}, ${p.y})`}
                    onMouseEnter={() => setHoverId(p.id)}
                    onMouseLeave={() => setHoverId(null)}
                    style={{ cursor: "pointer", opacity: lit ? 1 : 0.22, transition: "opacity 120ms var(--ease)" }}
                  >
                    <text
                      textAnchor={dir === "right" ? "start" : "end"}
                      dominantBaseline="middle"
                      style={{
                        fontSize: fs,
                        fontWeight: isRoot ? 700 : 400,
                        fill: isRoot || (connected != null && p.id === hoverId) ? "var(--accent)" : "var(--fg)",
                      }}
                    >
                      {p.word}
                    </text>
                    {!isRoot && (
                      <text
                        x={dir === "right" ? p.word.length * (fs * 0.55) + 6 : -(p.word.length * (fs * 0.55) + 6)}
                        textAnchor={dir === "right" ? "start" : "end"}
                        dominantBaseline="middle"
                        style={{ fontSize: 10, fill: "var(--fg-subtle)", fontFamily: "var(--font-mono)" }}
                      >
                        {p.count}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          )}
        </div>

        <div className="cx-coll-legend">
          <span className="cx-coll-leg-note">
            branches from {hits.length.toLocaleString()} concordance line{hits.length === 1 ? "" : "s"} · {leaves}{" "}
            distinct path{leaves === 1 ? "" : "s"} · text size ∝ frequency
            {result?.truncated ? " · (first 200 lines)" : ""}
          </span>
        </div>
      </div>
    </div>
  );
}
