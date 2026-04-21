// Top-level layout: left sidebar (corpora) + main pane (query bar + KWIC
// results). This is the surface Claude Design will re-skin. All state
// here is placeholder — real Tauri wiring lands once visual direction
// is locked in.

import { useState } from "react";
import { Sidebar } from "@/components/layout/Sidebar";
import { QueryBar } from "@/components/query/QueryBar";
import { KwicTable } from "@/components/results/KwicTable";
import type { CorpusMeta, KwicResult, QueryLayer } from "@/types";

export function App() {
  const [corpora] = useState<CorpusMeta[]>([]);
  const [activeCorpusId, setActiveCorpusId] = useState<string | null>(null);
  const [result, setResult] = useState<KwicResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleRun = async (params: { term: string; layer: QueryLayer }) => {
    // TODO: wire to runKwic(...) from @/lib/tauri
    setLoading(true);
    setResult({
      hits: [],
      elapsedMs: 0,
      truncated: false,
    });
    setLoading(false);
    void params;
  };

  return (
    <div className="flex h-full w-full">
      <Sidebar
        corpora={corpora}
        activeCorpusId={activeCorpusId}
        onSelect={setActiveCorpusId}
        onOpenCorpus={() => {
          /* TODO: openCorpus dialog */
        }}
        onBuildCorpus={() => {
          /* TODO: buildIndex dialog */
        }}
      />
      <main className="flex flex-1 flex-col bg-background">
        <QueryBar disabled={activeCorpusId === null} onRun={handleRun} />
        <div className="flex-1 overflow-hidden">
          <KwicTable result={result} loading={loading} />
        </div>
      </main>
    </div>
  );
}
