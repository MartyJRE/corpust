// Top-of-main query input. The tool's "address bar": pick layer + type a
// term + hit enter. Results stream into KwicTable below.

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import type { QueryLayer } from "@/types";

const LAYERS: { value: QueryLayer; label: string; hint: string }[] = [
  { value: "word", label: "word", hint: "surface form (case-insensitive)" },
  { value: "lemma", label: "lemma", hint: "dictionary form — requires annotation" },
  { value: "pos", label: "pos", hint: "POS tag — case-sensitive (e.g. NN, VBD)" },
];

export interface QueryBarProps {
  disabled?: boolean;
  onRun: (params: { term: string; layer: QueryLayer }) => void;
}

export function QueryBar({ disabled, onRun }: QueryBarProps) {
  const [term, setTerm] = useState("");
  const [layer, setLayer] = useState<QueryLayer>("word");

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!term.trim()) return;
    onRun({ term: term.trim(), layer });
  };

  return (
    <form
      onSubmit={submit}
      className="flex items-center gap-2 border-b border-border bg-card px-4 py-2"
    >
      <div className="flex gap-1 rounded-md border border-border p-0.5">
        {LAYERS.map((l) => (
          <button
            key={l.value}
            type="button"
            title={l.hint}
            onClick={() => setLayer(l.value)}
            className={`rounded px-2 py-1 font-mono text-xs transition-colors ${
              layer === l.value
                ? "bg-accent text-accent-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {l.label}
          </button>
        ))}
      </div>
      <Input
        value={term}
        onChange={(e) => setTerm(e.target.value)}
        placeholder={
          layer === "pos"
            ? "POS tag (e.g. NN, VBD, IN)…"
            : "term or regex…"
        }
        className="font-mono"
        disabled={disabled}
        autoFocus
      />
      <Button type="submit" size="sm" disabled={disabled || !term.trim()}>
        Run
      </Button>
    </form>
  );
}
