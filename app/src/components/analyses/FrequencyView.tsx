import { useMemo, useState } from "react";
import { DOC_FREQ, POS_FREQ, WORD_FREQ, makeDispersion } from "@/data";
import type { CorpusMeta, FreqBy, FreqRow } from "@/types";

export interface FrequencyViewProps {
  corpus: CorpusMeta;
  term: string;
}

export function FrequencyView({ corpus, term }: FrequencyViewProps) {
  const [by, setBy] = useState<FreqBy>("word");
  const dispersion = useMemo(() => makeDispersion(42), []);
  const freq: FreqRow[] = by === "pos" ? POS_FREQ : WORD_FREQ;
  const maxCount = freq[0].count;

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
            {freq.map((row) => {
              const w = (row.count / maxCount) * 100;
              const key = row.word ?? row.tag ?? "";
              return (
                <div key={key} className="cx-freq-bar-row">
                  <span className={`word ${by === "pos" ? "is-pos" : ""}`}>
                    {by === "pos" ? `${row.tag} · ${row.label}` : row.word}
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
            <div className="cx-card-meta">{dispersion.length} hits · 100 buckets</div>
          </div>
          <div className="cx-card-body">
            <div className="cx-disp">
              {dispersion.map((pct, i) => (
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
            {DOC_FREQ.slice(0, 8).map((d) => (
              <div
                key={d.doc}
                className="cx-freq-bar-row"
                style={{ gridTemplateColumns: "140px 1fr 50px 60px" }}
              >
                <span className="word" style={{ fontSize: 11 }}>
                  {d.doc}
                </span>
                <div
                  className="cx-freq-bar"
                  style={{ width: `${(d.hits / DOC_FREQ[0].hits) * 100}%` }}
                />
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
