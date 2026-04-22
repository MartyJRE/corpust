import { X } from "lucide-react";
import { useState } from "react";
import type { CorpusMeta } from "@/types";

export interface BuildDialogProps {
  open: boolean;
  onClose: () => void;
  onBuilt: (corpus: CorpusMeta) => void;
}

type Phase = "idle" | "reading" | "indexing" | "annotating" | "done";

const TOTAL_DOCS = 544;

export function BuildDialog({ open, onClose, onBuilt }: BuildDialogProps) {
  const [path, setPath] = useState("~/corpora/new-corpus");
  const [annotate, setAnnotate] = useState(true);
  const [name, setName] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState(0);
  const [docIdx, setDocIdx] = useState(0);

  if (!open) return null;

  const start = () => {
    setPhase("reading");
    setProgress(0.05);
    setTimeout(() => {
      setPhase("indexing");
      setProgress(0.25);
      setDocIdx(140);
    }, 500);
    setTimeout(() => {
      setPhase("indexing");
      setProgress(0.55);
      setDocIdx(298);
    }, 1100);
    if (annotate) {
      setTimeout(() => {
        setPhase("annotating");
        setProgress(0.78);
        setDocIdx(430);
      }, 1700);
    }
    setTimeout(() => {
      setPhase("indexing");
      setProgress(0.94);
      setDocIdx(512);
    }, 2200);
    setTimeout(() => {
      setPhase("done");
      setProgress(1);
      setDocIdx(TOTAL_DOCS);
      onBuilt({
        id: "new-" + Date.now(),
        kind: "mixed",
        name: name.trim() || path.split("/").pop() || "new corpus",
        indexPath: path + "/index",
        sourcePath: path,
        annotated: annotate,
        docCount: TOTAL_DOCS,
        tokenCount: 79_467_311,
        types: 412_908,
        avgDocLen: 146_083,
        builtAt: new Date().toISOString(),
        buildMs: annotate ? 211_000 : 17_500,
        languages: ["en"],
        tokeniser: "corpust-v0.6 · default",
        annotator: annotate ? "TreeTagger · english-utf8.par" : null,
        sizeOnDisk: annotate ? 612_000_000 : 188_000_000,
      });
    }, 2800);
  };

  const phaseMsg =
    phase === "reading"
      ? "reading documents…"
      : phase === "indexing"
        ? `indexing · ${docIdx} / ${TOTAL_DOCS} docs · ${Math.round(4_500_000 * (progress + 0.1)).toLocaleString()} wps`
        : phase === "annotating"
          ? `annotating · TreeTagger · ${docIdx} / ${TOTAL_DOCS} docs`
          : phase === "done"
            ? `built · ${annotate ? "3:31" : "17.5 s"} · 79,467,311 tokens`
            : "";

  return (
    <div className="cx-modal-backdrop" onClick={phase === "idle" ? onClose : undefined}>
      <div className="cx-modal" onClick={(e) => e.stopPropagation()}>
        <div className="cx-modal-head">
          <div>
            <h2 className="cx-modal-title">Build index</h2>
          </div>
          <button type="button" className="cx-btn cx-btn-ghost cx-btn-icon" onClick={onClose}>
            <X size={13} />
          </button>
        </div>
        <p className="cx-modal-desc">
          Read <span style={{ fontFamily: "var(--font-mono)", color: "var(--fg)" }}>.txt</span> files from a directory and build a Tantivy-backed corpus index.
        </p>

        <div className="cx-form-row">
          <label className="cx-form-label">
            Source directory <span className="cx-form-hint">recursive</span>
          </label>
          <input
            className="cx-input cx-input-mono"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            disabled={phase !== "idle"}
            style={{ paddingLeft: 12 }}
          />
        </div>
        <div className="cx-form-row">
          <label className="cx-form-label">
            Name <span className="cx-form-hint">defaults to folder name</span>
          </label>
          <input
            className="cx-input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={phase !== "idle"}
            placeholder="e.g. Gutenberg · EN"
            style={{ paddingLeft: 12, fontFamily: "var(--font-sans)" }}
          />
        </div>
        <div className="cx-form-row">
          <label className="cx-checkbox">
            <input
              type="checkbox"
              checked={annotate}
              onChange={(e) => setAnnotate(e.target.checked)}
              disabled={phase !== "idle"}
            />
            <span>
              Annotate with TreeTagger (lemma + POS)<span className="sub"> · ≈18× slower</span>
            </span>
          </label>
        </div>

        {phase !== "idle" && (
          <div className="cx-progress">
            <div className="cx-progress-head">
              <span>{phaseMsg}</span>
              <span>{Math.round(progress * 100)}%</span>
            </div>
            <div className="cx-progress-bar">
              <div style={{ width: `${progress * 100}%` }} />
            </div>
            <div className="cx-progress-meta">
              <span>eta {phase === "done" ? "—" : `${Math.max(1, Math.round((1 - progress) * (annotate ? 210 : 18)))} s`}</span>
              <span>peak mem {phase === "done" ? "428 MB" : `${Math.round(120 + progress * 300)} MB`}</span>
            </div>
          </div>
        )}

        <div className="cx-modal-foot">
          <button type="button" className="cx-btn cx-btn-outline" onClick={onClose}>
            {phase === "done" ? "Close" : "Cancel"}
          </button>
          {phase === "idle" && (
            <button type="button" className="cx-btn cx-btn-primary" onClick={start}>
              Build
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
