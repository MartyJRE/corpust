// Left sidebar: list of loaded corpora + "open another corpus" affordance.
// Placeholder shell — Claude Design will populate the visual treatment.

import type { CorpusMeta } from "@/types";
import { CorpusList } from "@/components/corpora/CorpusList";
import { Separator } from "@/components/ui/separator";

export interface SidebarProps {
  corpora: CorpusMeta[];
  activeCorpusId: string | null;
  onSelect: (id: string) => void;
  onOpenCorpus: () => void;
  onBuildCorpus: () => void;
}

export function Sidebar(props: SidebarProps) {
  return (
    <aside className="flex h-full w-64 flex-col border-r border-border bg-card">
      <div className="flex items-center gap-2 px-4 py-3">
        <div className="font-serif text-lg tracking-tight">corpust</div>
      </div>
      <Separator />
      <div className="flex-1 overflow-y-auto">
        <CorpusList
          corpora={props.corpora}
          activeId={props.activeCorpusId}
          onSelect={props.onSelect}
        />
      </div>
      <Separator />
      <div className="flex gap-2 p-3">
        {/* Wiring for Claude Design to fill in — buttons for "open" + "build" */}
        <button
          type="button"
          onClick={props.onOpenCorpus}
          className="flex-1 text-xs text-muted-foreground hover:text-foreground"
        >
          open
        </button>
        <button
          type="button"
          onClick={props.onBuildCorpus}
          className="flex-1 text-xs text-muted-foreground hover:text-foreground"
        >
          build
        </button>
      </div>
    </aside>
  );
}
