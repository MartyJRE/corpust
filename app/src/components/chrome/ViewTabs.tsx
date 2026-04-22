import { BarChart3, Network, Table2 } from "lucide-react";
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
    { id: "kwic", label: "concordance", Icon: Table2, count: result ? result.hits.length : null },
    { id: "coll", label: "collocations", Icon: Network, count: 14 },
    { id: "freq", label: "frequency", Icon: BarChart3, count: null },
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
