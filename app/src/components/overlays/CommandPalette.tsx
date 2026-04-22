import { BarChart3, Download, Folder, Hammer, Info, Layers, Network, Sparkles, Table2, X } from "lucide-react";
import type { ComponentType } from "react";
import { useEffect, useMemo, useState } from "react";

type IconT = ComponentType<{ size?: number }>;

export interface CommandDef {
  group: string;
  id: string;
  label: string;
  icon: IconT;
  kbd?: string;
}

const COMMANDS: CommandDef[] = [
  { group: "Corpora", id: "build", label: "Build index from folder…", icon: Hammer, kbd: "⌘B" },
  { group: "Corpora", id: "open", label: "Open existing corpus…", icon: Folder, kbd: "⌘O" },
  { group: "Corpora", id: "detail", label: "Show corpus metadata", icon: Info },
  { group: "Query", id: "layer-word", label: "Switch layer: word", icon: Layers, kbd: "⌘1" },
  { group: "Query", id: "layer-lemma", label: "Switch layer: lemma", icon: Layers, kbd: "⌘2" },
  { group: "Query", id: "layer-pos", label: "Switch layer: pos", icon: Layers, kbd: "⌘3" },
  { group: "Query", id: "clear", label: "Clear query", icon: X, kbd: "⌘⌫" },
  { group: "View", id: "view-kwic", label: "Go to concordance", icon: Table2 },
  { group: "View", id: "view-coll", label: "Go to collocations", icon: Network },
  { group: "View", id: "view-freq", label: "Go to frequency", icon: BarChart3 },
  { group: "Results", id: "export-csv", label: "Export concordance as CSV", icon: Download, kbd: "⌘E" },
  { group: "Results", id: "export-json", label: "Export concordance as JSON", icon: Download },
];

export interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  onRun: (cmd: CommandDef) => void;
}

export function CommandPalette({ open, onClose, onRun }: CommandPaletteProps) {
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);

  const filtered = useMemo(() => {
    const n = q.trim().toLowerCase();
    if (!n) return COMMANDS;
    return COMMANDS.filter((c) => c.label.toLowerCase().includes(n) || c.id.includes(n));
  }, [q]);

  useEffect(() => {
    setSel(0);
  }, [q]);

  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowDown") {
        e.preventDefault();
        setSel((s) => Math.min(filtered.length - 1, s + 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSel((s) => Math.max(0, s - 1));
      } else if (e.key === "Enter") {
        e.preventDefault();
        const c = filtered[sel];
        if (c) {
          onRun(c);
          onClose();
        }
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open, sel, filtered, onClose, onRun]);

  if (!open) return null;

  const groups: Record<string, (CommandDef & { idx: number })[]> = {};
  filtered.forEach((c, i) => {
    const arr = groups[c.group] ?? (groups[c.group] = []);
    arr.push({ ...c, idx: i });
  });

  return (
    <div className="cx-palette-backdrop" onClick={onClose}>
      <div className="cx-palette" onClick={(e) => e.stopPropagation()}>
        <div className="cx-palette-search">
          <span className="cx-palette-prompt">→</span>
          <input
            autoFocus
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="type a command or search…"
            className="cx-palette-input"
          />
          <span className="cx-kbd">esc</span>
        </div>
        <div className="cx-palette-body">
          {filtered.length === 0 ? (
            <div className="cx-palette-empty">No commands match.</div>
          ) : (
            Object.entries(groups).map(([group, items]) => (
              <div key={group} className="cx-palette-group">
                <div className="cx-palette-grp-label">{group}</div>
                {items.map((c) => {
                  const IconC = c.icon ?? Sparkles;
                  return (
                    <div
                      key={c.id}
                      className={`cx-palette-item ${c.idx === sel ? "is-sel" : ""}`}
                      onMouseEnter={() => setSel(c.idx)}
                      onClick={() => {
                        onRun(c);
                        onClose();
                      }}
                    >
                      <div className="left">
                        <IconC size={13} />
                        <span>{c.label}</span>
                      </div>
                      {c.kbd && <span className="cx-kbd">{c.kbd}</span>}
                    </div>
                  );
                })}
              </div>
            ))
          )}
        </div>
        <div className="cx-palette-foot">
          <span>
            {filtered.length} command{filtered.length === 1 ? "" : "s"}
          </span>
          <div className="right">
            <span className="k">
              <span className="cx-kbd">↑↓</span> navigate
            </span>
            <span className="k">
              <span className="cx-kbd">↵</span> run
            </span>
            <span className="k">
              <span className="cx-kbd">esc</span> close
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
