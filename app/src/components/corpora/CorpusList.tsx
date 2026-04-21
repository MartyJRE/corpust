// Vertical list of loaded corpora. Highlights the active one.

import type { CorpusMeta } from "@/types";
import { cn, formatNumber } from "@/lib/utils";

export interface CorpusListProps {
  corpora: CorpusMeta[];
  activeId: string | null;
  onSelect: (id: string) => void;
}

export function CorpusList({ corpora, activeId, onSelect }: CorpusListProps) {
  if (corpora.length === 0) {
    return (
      <div className="p-4 text-xs text-muted-foreground">
        No corpora yet. Open or build one to get started.
      </div>
    );
  }

  return (
    <ul className="flex flex-col">
      {corpora.map((c) => (
        <li key={c.id}>
          <button
            type="button"
            onClick={() => onSelect(c.id)}
            className={cn(
              "flex w-full flex-col items-start gap-0.5 px-4 py-2 text-left text-sm hover:bg-accent/50",
              c.id === activeId && "bg-accent text-accent-foreground",
            )}
          >
            <span className="font-medium">{c.name}</span>
            <span className="text-xs text-muted-foreground">
              {formatNumber(c.docCount)} docs ·{" "}
              {formatNumber(c.tokenCount)} tokens
              {c.annotated ? " · annotated" : ""}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}
