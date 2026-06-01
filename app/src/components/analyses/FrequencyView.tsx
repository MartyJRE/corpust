import { useEffect, useMemo, useState } from "react";
import { DOC_FREQ, POS_FREQ, WORD_FREQ, makeDispersion } from "@/data";
import {
  type FreqResultRow,
  type TermDistResult,
  hasLiveData,
  runFrequencies,
  runTermDistribution,
} from "@/lib/tauri";
import { basename } from "@/lib/utils";
import type { CorpusMeta, FreqBy } from "@/types";

export interface FrequencyViewProps {
  corpus: CorpusMeta;
  term: string;
}

/** How many bars to show in the top-terms list, and how many buckets to
 *  split the dispersion axis into. */
const TOP_N = 12;
const BUCKETS = 100;

interface FreqBarRow {
  primary: string;
  secondary?: string;
  count: number;
  pct: number;
}

interface DocFreqRow {
  doc: string;
  hits: number;
  per1m: number;
}

export function FrequencyView({ corpus, term }: FrequencyViewProps) {
  const [by, setBy] = useState<FreqBy>("word");
  const [liveFreq, setLiveFreq] = useState<FreqResultRow[] | null>(null);
  const [liveDist, setLiveDist] = useState<TermDistResult | null>(null);

  // Corpus-wide term frequencies on the active layer; refetch when the
  // corpus or the word/POS toggle changes. Fixtures fall through below.
  useEffect(() => {
    setLiveFreq(null);
    if (!hasLiveData(corpus.id)) return;
    let cancelled = false;
    runFrequencies({ corpusId: corpus.id, layer: by, limit: TOP_N })
      .then((r) => {
        if (!cancelled) setLiveFreq(r.rows);
      })
      .catch((e) => console.error("runFrequencies failed:", e));
    return () => {
      cancelled = true;
    };
  }, [corpus.id, by]);

  // Per-document hit counts + dispersion for the queried term.
  useEffect(() => {
    setLiveDist(null);
    if (!hasLiveData(corpus.id) || !term.trim()) return;
    let cancelled = false;
    runTermDistribution({ corpusId: corpus.id, term: term.trim(), layer: by, buckets: BUCKETS })
      .then((r) => {
        if (!cancelled) setLiveDist(r);
      })
      .catch((e) => console.error("runTermDistribution failed:", e));
    return () => {
      cancelled = true;
    };
  }, [corpus.id, term, by]);

  const fixtureDispersion = useMemo(() => makeDispersion(42), []);

  // --- Normalise live or fixture data into uniform render rows. ---

  const freqRows: FreqBarRow[] = liveFreq
    ? liveFreq.map((r) => ({ primary: r.term, count: r.count, pct: r.pct }))
    : (by === "pos" ? POS_FREQ : WORD_FREQ).map((r) => ({
        primary: (by === "pos" ? r.tag : r.word) ?? "",
        secondary: by === "pos" ? r.label : undefined,
        count: r.count,
        pct: r.pct,
      }));
  const maxCount = freqRows.length ? freqRows[0].count : 1;

  const docFreq: DocFreqRow[] = liveDist
    ? liveDist.docCounts.map((d) => ({
        doc: basename(d.path),
        hits: d.hits,
        per1m: d.tokenCount > 0 ? (d.hits / d.tokenCount) * 1_000_000 : 0,
      }))
    : DOC_FREQ;
  const maxDocHits = docFreq.length ? docFreq[0].hits : 1;

  // Barcode dispersion: a mark at each non-empty bucket along the axis.
  const dispMarks: number[] = liveDist
    ? liveDist.dispersion.flatMap((c, i) =>
        c > 0 ? [(i / liveDist.dispersion.length) * 100] : [],
      )
    : fixtureDispersion;
  const dispHits = liveDist ? liveDist.totalHits : fixtureDispersion.length;
  const dispBuckets = liveDist ? liveDist.dispersion.length : BUCKETS;

  return (
    <div className="cx-freq-wrap">
      <div className="cx-freq-head">
        <h2 className="cx-freq-title">
          frequency{" "}
          <span style={{ color: "var(--fg-muted)", fontSize: 14 }}>· {corpus.name}</span>
        </h2>
        <div className="cx-coll-segbtn">
          <button type="button" className={by === "word" ? "is-on" : ""} onClick={() => setBy("word")}>
            by word
          </button>
          <button
            type="button"
            className={by === "pos" ? "is-on" : ""}
            onClick={() => setBy("pos")}
            disabled={!corpus.annotated}
          >
            by POS
          </button>
        </div>
      </div>

      <div className="cx-freq-grid">
        <div className="cx-card">
          <div className="cx-card-head">
            <div className="cx-card-title">top {by === "pos" ? "POS tags" : "wordforms"}</div>
            <div className="cx-card-meta">n = {corpus.tokenCount.toLocaleString()} tokens</div>
          </div>
          <div className="cx-card-body">
            {freqRows.map((row) => {
              const w = (row.count / maxCount) * 100;
              return (
                <div key={row.primary} className="cx-freq-bar-row">
                  <span className={`word ${by === "pos" ? "is-pos" : ""}`}>
                    {row.secondary ? `${row.primary} · ${row.secondary}` : row.primary}
                  </span>
                  <div className="cx-freq-bar" style={{ width: `${w}%` }} />
                  <span className="count">{row.count.toLocaleString()}</span>
                  <span className="pct">{row.pct.toFixed(2)}%</span>
                </div>
              );
            })}
          </div>
        </div>

        <div className="cx-card">
          <div className="cx-card-head">
            <div className="cx-card-title">
              dispersion · <span style={{ fontFamily: "var(--font-mono)" }}>"{term}"</span>
            </div>
            <div className="cx-card-meta">
              {dispHits.toLocaleString()} hits · {dispBuckets} buckets
            </div>
          </div>
          <div className="cx-card-body">
            <div className="cx-disp">
              {dispMarks.map((pct, i) => (
                <div key={i} className="cx-disp-mark" style={{ left: `${pct}%` }} />
              ))}
            </div>
            <div className="cx-disp-axis">
              <span>doc 0</span>
              <span>0.25</span>
              <span>0.5</span>
              <span>0.75</span>
              <span>doc {corpus.docCount}</span>
            </div>

            <div className="cx-section-h" style={{ marginTop: 24 }}>
              <span>top documents</span>
              <span className="sub">hits · per 1M tokens</span>
            </div>
            {docFreq.slice(0, 8).map((d) => (
              <div
                key={d.doc}
                className="cx-freq-bar-row"
                style={{ gridTemplateColumns: "140px 1fr 50px 60px" }}
              >
                <span className="word" style={{ fontSize: 11 }}>
                  {d.doc}
                </span>
                <div className="cx-freq-bar" style={{ width: `${(d.hits / maxDocHits) * 100}%` }} />
                <span className="count">{d.hits}</span>
                <span className="pct">{d.per1m.toFixed(1)}/M</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
