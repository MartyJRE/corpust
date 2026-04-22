import { BookOpen, Folder, Hammer, Layers, Library, Newspaper, Plus, Scale } from "lucide-react";
import type { ComponentType } from "react";
import type { CorpusKind, CorpusMeta, RecentQuery } from "@/types";
import { Wordmark } from "./Wordmark";

const KIND_ICON: Record<CorpusKind, ComponentType<{ size?: number; className?: string }>> = {
  literary: BookOpen,
  legal: Scale,
  news: Newspaper,
  mixed: Layers,
};

function CorpusRow({
  corpus,
  active,
  onSelect,
}: {
  corpus: CorpusMeta;
  active: boolean;
  onSelect: (id: string) => void;
}) {
  const IconC = KIND_ICON[corpus.kind] ?? Library;
  const tokensM = corpus.tokenCount / 1_000_000;
  return (
    <div
      className={`cx-corpus-row ${active ? "is-active" : ""}`}
      role="button"
      tabIndex={0}
      onClick={() => onSelect(corpus.id)}
      onKeyDown={(e) => {
        if (e.key === "Enter") onSelect(corpus.id);
      }}
    >
      <div className="cx-corpus-name">
        <IconC size={12} className="cx-corpus-icon" />
        <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {corpus.name}
        </span>
        {corpus.annotated && <span className="cx-annot">annot</span>}
      </div>
      <div className="cx-corpus-meta">
        <span>{corpus.docCount.toLocaleString()} docs</span>
        <span className="dot">·</span>
        <span>{tokensM.toFixed(tokensM < 10 ? 2 : 1)}M tok</span>
      </div>
    </div>
  );
}

function RecentQueryRow({ q, onRun }: { q: RecentQuery; onRun: (q: RecentQuery) => void }) {
  return (
    <button type="button" className="cx-recent-row" onClick={() => onRun(q)}>
      <span className={`cx-recent-layer cx-layer-chip cx-layer-${q.layer}`}>{q.layer}</span>
      <span className="cx-recent-term">{q.term}</span>
      <span className="cx-recent-count">{q.hits.toLocaleString()}</span>
    </button>
  );
}

export interface SidebarProps {
  corpora: CorpusMeta[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onOpen: () => void;
  onBuild: () => void;
  recent: RecentQuery[];
  onRunRecent: (q: RecentQuery) => void;
}

export function Sidebar({
  corpora,
  activeId,
  onSelect,
  onOpen,
  onBuild,
  recent,
  onRunRecent,
}: SidebarProps) {
  return (
    <aside className="cx-sidebar">
      <div className="cx-sidebar-head">
        <div>
          <Wordmark />
          <div className="cx-wordmark-sub">v0.6 · pre-alpha</div>
        </div>
      </div>
      <div className="cx-section-label">
        <span>
          Corpora <span className="count">{corpora.length}</span>
        </span>
        <button type="button" className="add-btn" onClick={onBuild} title="Build new index (⌘B)">
          <Plus size={12} />
        </button>
      </div>
      <div className="cx-sidebar-list flex-grow">
        {corpora.length === 0 ? (
          <div style={{ padding: "12px 14px", fontSize: 12, color: "var(--fg-muted)" }}>
            No corpora yet. Open or build one to get started.
          </div>
        ) : (
          corpora.map((c) => (
            <CorpusRow key={c.id} corpus={c} active={c.id === activeId} onSelect={onSelect} />
          ))
        )}
      </div>

      <div className="cx-section-label">
        <span>
          Recent <span className="count">{recent.length}</span>
        </span>
      </div>
      <div className="cx-sidebar-list" style={{ paddingBottom: 6 }}>
        {recent.slice(0, 4).map((q) => (
          <RecentQueryRow key={q.id} q={q} onRun={onRunRecent} />
        ))}
      </div>

      <div className="cx-sidebar-foot">
        <div className="cx-sidebar-foot-row">
          <button type="button" className="cx-btn cx-btn-ghost grow" onClick={onOpen}>
            <Folder size={12} />
            open
          </button>
          <button type="button" className="cx-btn cx-btn-ghost grow" onClick={onBuild}>
            <Hammer size={12} />
            build
          </button>
        </div>
      </div>
    </aside>
  );
}
