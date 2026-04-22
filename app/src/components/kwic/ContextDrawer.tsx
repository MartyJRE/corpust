// Serif-bodied context drawer — the user switches from "scanning" to
// "reading" here, so typography pivots from JetBrains Mono to Source
// Serif 4. 420px wide, slides in 180ms.

import { ArrowLeft, ArrowRight, FileText, X } from "lucide-react";
import { Fragment } from "react";
import { EXPANDED } from "@/data";
import type { CorpusMeta, KwicHit } from "@/types";

export interface ContextDrawerProps {
  hit: KwicHit;
  corpus: CorpusMeta;
  onClose: () => void;
  onPrev: () => void;
  onNext: () => void;
}

export function ContextDrawer({ hit, corpus, onClose, onPrev, onNext }: ContextDrawerProps) {
  const expanded = EXPANDED[`${corpus.id}|${hit.docId}|${hit.pos}`] ?? {
    before:
      "Context before the hit would appear here in three sentences, loaded lazily from the underlying document on the Rust side. The drawer renders in Source Serif 4 because at this point the user is reading rather than scanning; monospace lives in the table, serif lives here. ",
    match: `${hit.left} ${hit.hit} ${hit.right}`,
    after:
      ". After the hit, three more sentences of context; the drawer is scrollable if the surrounding paragraph is long. ←/→ move between hits without closing.",
    docTitle: hit.docId.replace(".txt", ""),
    docMeta: `position ${hit.pos.toLocaleString()}`,
  };

  const parts = expanded.match.split(hit.hit);

  return (
    <aside className="cx-drawer">
      <div className="cx-drawer-head">
        <div className="cx-drawer-title">
          <FileText size={12} />
          <span style={{ color: "var(--fg)" }}>{expanded.docTitle}</span>
          <span className="pos">·</span>
          <span className="pos">pos {hit.pos.toLocaleString()}</span>
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          <button type="button" className="cx-btn cx-btn-ghost cx-btn-icon" onClick={onPrev} title="Previous hit (k)">
            <ArrowLeft size={13} />
          </button>
          <button type="button" className="cx-btn cx-btn-ghost cx-btn-icon" onClick={onNext} title="Next hit (j)">
            <ArrowRight size={13} />
          </button>
          <button type="button" className="cx-btn cx-btn-ghost cx-btn-icon" onClick={onClose} title="Close (esc)">
            <X size={13} />
          </button>
        </div>
      </div>
      <div className="cx-drawer-body">
        <span style={{ color: "var(--fg-muted)" }}>{expanded.before}</span>
        <span>
          {parts.map((part, i) => (
            <Fragment key={i}>
              {part}
              {i < parts.length - 1 && <span className="cx-drawer-hit">{hit.hit}</span>}
            </Fragment>
          ))}
        </span>
        <span style={{ color: "var(--fg-muted)" }}>{expanded.after}</span>
      </div>
      <div className="cx-drawer-foot">
        <span>{expanded.docMeta}</span>
        <div className="cx-drawer-nav">
          <span className="cx-kbd">j</span>
          <span>/</span>
          <span className="cx-kbd">k</span>
          <span style={{ color: "var(--fg-subtle)", margin: "0 4px" }}>·</span>
          <span className="cx-kbd">esc</span>
        </div>
      </div>
    </aside>
  );
}
