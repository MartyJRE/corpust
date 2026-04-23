import { FolderOpen, X } from "lucide-react";
import { useState } from "react";
import type { CorpusMeta } from "@/types";
import { buildIndex, inTauri } from "@/lib/tauri";

async function pickDirectory(): Promise<string | null> {
  const { open } = await import("@tauri-apps/plugin-dialog");
  const selected = await open({ directory: true, multiple: false });
  return typeof selected === "string" ? selected : null;
}

export interface BuildDialogProps {
  open: boolean;
  onClose: () => void;
  onBuilt: (corpus: CorpusMeta) => void;
}

type Phase = "idle" | "reading" | "indexing" | "annotating" | "done" | "failed";

const TOTAL_DOCS = 544;

export function BuildDialog({ open, onClose, onBuilt }: BuildDialogProps) {
  const [path, setPath] = useState("");
  const [annotate, setAnnotate] = useState(true);
  const [name, setName] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState(0);
  const [docIdx, setDocIdx] = useState(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  if (!open) return null;

  // Default the output path to `<source>.index/` so the user doesn't
  // have to think about a second location. Real apps will want a
  // settings-managed `<data_dir>/corpust/corpora/` — tracked in
  // issue #1.
  const outPath = path.replace(/\/+$/, "") + ".index";

  const runRealBuild = async () => {
    setErrorMsg(null);
    setPhase("reading");
    setProgress(0.05);
    setDocIdx(0);
    // Cheap animated progress while the (synchronous) Rust call
    // runs. Real event-driven progress is a follow-up.
    const ticker = setInterval(() => {
      setProgress((p) => Math.min(0.92, p + 0.01));
      setDocIdx((n) => Math.min(TOTAL_DOCS - 10, n + 6));
      setPhase((ph) => (ph === "reading" ? "indexing" : ph));
    }, 120);
    try {
      const meta = await buildIndex({
        sourcePath: path,
        outPath,
        annotate,
        name: name.trim() || undefined,
      });
      clearInterval(ticker);
      setPhase("done");
      setProgress(1);
      setDocIdx(meta.docCount || TOTAL_DOCS);
      onBuilt(meta);
    } catch (e) {
      clearInterval(ticker);
      setPhase("failed");
      setErrorMsg(String(e));
    }
  };

  const runFakeBuild = () => {
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
        indexPath: outPath,
        sourcePath: path,
        annotated: annotate,
        docCount: TOTAL_DOCS,
        tokenCount: 79_467_311,
        types: 412_908,
        avgDocLen: 146_083,
        builtAt: new Date().toISOString(),
        buildMs: annotate ? 55_500 : 17_500,
        languages: ["en"],
        tokeniser: "corpust-v0.6 · default",
        annotator: annotate ? "treetagger-rs-english" : null,
        sizeOnDisk: annotate ? 1_100_000_000 : 188_000_000,
      });
    }, 2800);
  };

  const start = () => {
    if (inTauri()) {
      void runRealBuild();
    } else {
      runFakeBuild();
    }
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
          <div style={{ display: "flex", gap: 8 }}>
            <input
              className="cx-input cx-input-mono"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              disabled={phase !== "idle"}
              placeholder={inTauri() ? "click browse, or paste an absolute path" : "/path/to/corpus"}
              style={{ paddingLeft: 12, flex: 1 }}
            />
            {inTauri() && (
              <button
                type="button"
                className="cx-btn cx-btn-outline"
                onClick={async () => {
                  const picked = await pickDirectory();
                  if (picked) setPath(picked);
                }}
                disabled={phase !== "idle"}
                title="pick a directory"
              >
                <FolderOpen size={13} />
                <span style={{ marginLeft: 6 }}>browse</span>
              </button>
            )}
          </div>
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
              <span>{phase === "failed" ? "build failed" : phaseMsg}</span>
              <span>{Math.round(progress * 100)}%</span>
            </div>
            <div className="cx-progress-bar">
              <div style={{ width: `${progress * 100}%` }} />
            </div>
            <div className="cx-progress-meta">
              <span>eta {phase === "done" || phase === "failed" ? "—" : `${Math.max(1, Math.round((1 - progress) * (annotate ? 55 : 18)))} s`}</span>
              <span>peak mem {phase === "done" ? "428 MB" : `${Math.round(120 + progress * 300)} MB`}</span>
            </div>
            {errorMsg && phase === "failed" && (
              <div className="cx-progress-meta" style={{ color: "var(--danger, #c33)", marginTop: 8 }}>
                <span style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>{errorMsg}</span>
              </div>
            )}
          </div>
        )}

        <div className="cx-modal-foot">
          <button type="button" className="cx-btn cx-btn-outline" onClick={onClose}>
            {phase === "done" ? "Close" : "Cancel"}
          </button>
          {phase === "idle" && (
            <button
              type="button"
              className="cx-btn cx-btn-primary"
              onClick={start}
              disabled={!path.trim()}
            >
              Build
            </button>
          )}
          {phase === "failed" && (
            <button
              type="button"
              className="cx-btn cx-btn-primary"
              onClick={() => {
                setPhase("idle");
                setErrorMsg(null);
                setProgress(0);
              }}
            >
              Retry
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
