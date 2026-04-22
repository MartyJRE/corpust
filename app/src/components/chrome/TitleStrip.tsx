import { Library, Search, Settings } from "lucide-react";
import { Wordmark } from "./Wordmark";
import type { MainView } from "@/types";

export interface TitleStripProps {
  view: MainView;
  onView: (v: MainView) => void;
}

const TABS: { id: MainView; label: string; Icon: typeof Search }[] = [
  { id: "search", label: "search", Icon: Search },
  { id: "corpus", label: "corpus", Icon: Library },
  { id: "settings", label: "settings", Icon: Settings },
];

export function TitleStrip({ view, onView }: TitleStripProps) {
  return (
    <div className="cx-titlestrip">
      <Wordmark />
      <div className="cx-ts-tabs" style={{ marginLeft: 14 }}>
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`cx-ts-tab ${view === t.id ? "is-on" : ""}`}
            onClick={() => onView(t.id)}
          >
            <t.Icon size={12} />
            {t.label}
          </button>
        ))}
      </div>
      <div className="cx-ts-spacer" />
    </div>
  );
}
