import { Plus, Search, X } from "lucide-react";
import { type FormEvent, useEffect, useState } from "react";
import { clearDimension, filterChips, normalizeFilter } from "@/lib/filter";
import type { DocFilter, QueryLayer } from "@/types";

const LAYERS: { value: QueryLayer; label: string; hint: string }[] = [
  { value: "word", label: "word", hint: "surface form · case-insensitive" },
  { value: "lemma", label: "lemma", hint: "dictionary form · requires annotation" },
  { value: "pos", label: "pos", hint: "POS tag · case-sensitive (NN, VBD, …)" },
];

export interface QueryBarProps {
  layer: QueryLayer;
  term: string;
  onLayer: (l: QueryLayer) => void;
  onTerm: (t: string) => void;
  onRun: () => void;
  disabled?: boolean;
  annotated?: boolean;
  onOpenPalette: () => void;
  /** Active document-metadata filter (raw, un-normalized). */
  filter: DocFilter;
  onFilterChange: (f: DocFilter) => void;
  /** Whether the active corpus supports filtering (real, backend-backed).
   *  Fixtures / preview have no queryable metadata, so the control hides. */
  filterable: boolean;
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
  filter,
  onFilterChange,
  filterable,
}: QueryBarProps) {
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (term.trim()) onRun();
  };

  const chips = filterChips(filter);

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

      {filterable && (
        <>
          {chips.map((c) => (
            <span
              key={c.key}
              className="cx-filter-chip is-on"
              onClick={() => onFilterChange(clearDimension(filter, c.key))}
              title="Remove filter"
              role="button"
              tabIndex={0}
            >
              {c.label}
              <span className="x">
                <X size={10} />
              </span>
            </span>
          ))}
          <FilterEditor filter={filter} onChange={onFilterChange} />
        </>
      )}

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

/** A small popover to add/edit metadata filters. Edits a local draft and
 *  commits on Apply, so each keystroke doesn't refetch every query. */
function FilterEditor({ filter, onChange }: { filter: DocFilter; onChange: (f: DocFilter) => void }) {
  const [open, setOpen] = useState(false);
  const [yearMin, setYearMin] = useState("");
  const [yearMax, setYearMax] = useState("");
  const [author, setAuthor] = useState("");
  const [path, setPath] = useState("");

  // Seed the draft from the active filter whenever the popover opens.
  useEffect(() => {
    if (!open) return;
    setYearMin(filter.yearMin != null ? String(filter.yearMin) : "");
    setYearMax(filter.yearMax != null ? String(filter.yearMax) : "");
    setAuthor(filter.author ?? "");
    setPath(filter.path ?? "");
  }, [open, filter]);

  // Close on Escape while open.
  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open]);

  const apply = () => {
    const draft: DocFilter = {
      yearMin: yearMin.trim() ? Number(yearMin) : undefined,
      yearMax: yearMax.trim() ? Number(yearMax) : undefined,
      author: author.trim() || undefined,
      path: path.trim() || undefined,
    };
    onChange(normalizeFilter(draft) ?? {});
    setOpen(false);
  };

  return (
    <span className="cx-filter-wrap">
      <button
        type="button"
        className="cx-filter-chip"
        title="Filter by document metadata"
        onClick={() => setOpen((o) => !o)}
      >
        <Plus size={10} />
        filter
      </button>
      {open && (
        <>
          <div className="cx-filter-backdrop" onClick={() => setOpen(false)} />
          <div className="cx-filter-pop" onClick={(e) => e.stopPropagation()}>
            <div className="cx-filter-pop-title">Filter documents</div>
            <label className="cx-filter-row">
              <span>year</span>
              <span className="cx-filter-range">
                <input
                  type="number"
                  className="cx-filter-input"
                  placeholder="from"
                  value={yearMin}
                  onChange={(e) => setYearMin(e.target.value)}
                />
                <span className="cx-filter-dash">–</span>
                <input
                  type="number"
                  className="cx-filter-input"
                  placeholder="to"
                  value={yearMax}
                  onChange={(e) => setYearMax(e.target.value)}
                />
              </span>
            </label>
            <label className="cx-filter-row">
              <span>author</span>
              <input
                type="text"
                className="cx-filter-input wide"
                placeholder="contains…"
                value={author}
                onChange={(e) => setAuthor(e.target.value)}
                spellCheck={false}
              />
            </label>
            <label className="cx-filter-row">
              <span>path</span>
              <input
                type="text"
                className="cx-filter-input wide"
                placeholder="contains…"
                value={path}
                onChange={(e) => setPath(e.target.value)}
                spellCheck={false}
              />
            </label>
            <div className="cx-filter-actions">
              <button
                type="button"
                className="cx-btn cx-btn-outline"
                onClick={() => {
                  onChange({});
                  setOpen(false);
                }}
              >
                Clear
              </button>
              <button type="button" className="cx-btn cx-btn-primary" onClick={apply}>
                Apply
              </button>
            </div>
          </div>
        </>
      )}
    </span>
  );
}
