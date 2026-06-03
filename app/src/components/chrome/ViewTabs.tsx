import { BarChart3, Grid3x3, LineChart, ListTree, Network, Table2 } from "lucide-react";
import type { KwicResult, SubView } from "@/types";

export interface ViewTabsProps {
  view: SubView;
  onView: (v: SubView) => void;
  result: KwicResult | null;
}

export function ViewTabs({ view, onView, result }: ViewTabsProps) {
  const tabs: {
    id: SubView;
    label: string;
    Icon: typeof Table2;
    count: number | null;
  }[] = [
    { id: "kwic", label: "concordance", Icon: Table2, count: result ? result.total : null },
    { id: "coll", label: "collocations", Icon: Network, count: null },
    { id: "freq", label: "frequency", Icon: BarChart3, count: null },
    { id: "tree", label: "word tree", Icon: ListTree, count: null },
    { id: "dist", label: "distance", Icon: Grid3x3, count: null },
    { id: "time", label: "over time", Icon: LineChart, count: null },
  ];
  return (
    <div className="cx-viewtabs">
      {tabs.map((t) => (
        <button
          key={t.id}
          type="button"
          className={`cx-viewtab ${view === t.id ? "is-on" : ""}`}
          onClick={() => onView(t.id)}
        >
          <t.Icon size={13} />
          {t.label}
          {t.count != null && <span className="count">{t.count.toLocaleString()}</span>}
        </button>
      ))}
    </div>
  );
}
