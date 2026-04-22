import { Plus, Search, X } from "lucide-react";
import type { FormEvent } from "react";
import type { QueryLayer } from "@/types";

const LAYERS: { value: QueryLayer; label: string; hint: string }[] = [
  { value: "word", label: "word", hint: "surface form · case-insensitive" },
  { value: "lemma", label: "lemma", hint: "dictionary form · requires annotation" },
  { value: "pos", label: "pos", hint: "POS tag · case-sensitive (NN, VBD, …)" },
];

export interface Filter {
  key: string;
  label: string;
}

export interface QueryBarProps {
  layer: QueryLayer;
  term: string;
  onLayer: (l: QueryLayer) => void;
  onTerm: (t: string) => void;
  onRun: () => void;
  disabled?: boolean;
  annotated?: boolean;
  onOpenPalette: () => void;
  filters: Filter[];
  onRemoveFilter: (k: string) => void;
}

export function QueryBar({
  layer,
  term,
  onLayer,
  onTerm,
  onRun,
  disabled,
  annotated,
  onOpenPalette,
  filters,
  onRemoveFilter,
}: QueryBarProps) {
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (term.trim()) onRun();
  };

  return (
    <form onSubmit={submit} className="cx-querybar">
      <div className="cx-layer-toggle" title="Linguistic query layer">
        {LAYERS.map((l) => {
          const locked = l.value !== "word" && !annotated;
          return (
            <button
              key={l.value}
              type="button"
              title={l.hint}
              onClick={() => onLayer(l.value)}
              disabled={locked}
              className={`cx-layer cx-layer-${l.value} ${layer === l.value ? "is-on" : ""}`}
            >
              {l.label}
            </button>
          );
        })}
      </div>

      <div className="cx-input-wrap">
        <span className="cx-input-icon">
          <Search size={14} />
        </span>
        <input
          className="cx-input cx-input-mono"
          value={term}
          onChange={(e) => onTerm(e.target.value)}
          placeholder={
            layer === "pos"
              ? "POS tag (e.g. NN, VBD, IN)…"
              : layer === "lemma"
                ? "lemma (e.g. go, be, run)…"
                : "term or regex…"
          }
          disabled={disabled}
          spellCheck={false}
        />
        <div className="cx-input-suffix">{term && <span>{layer === "pos" ? "exact" : "regex ok"}</span>}</div>
      </div>

      {filters.map((f) => (
        <span
          key={f.key}
          className="cx-filter-chip is-on"
          onClick={() => onRemoveFilter(f.key)}
          role="button"
          tabIndex={0}
        >
          {f.label}
          <span className="x">
            <X size={10} />
          </span>
        </span>
      ))}
      <button type="button" className="cx-filter-chip" title="Add a metadata filter">
        <Plus size={10} />
        filter
      </button>

      <button type="submit" className="cx-btn cx-btn-primary" disabled={disabled || !term.trim()}>
        Run
      </button>
      <button
        type="button"
        className="cx-btn cx-btn-outline cx-btn-icon"
        onClick={onOpenPalette}
        title="Command palette (⌘K)"
      >
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--fg-muted)" }}>⌘K</span>
      </button>
    </form>
  );
}
