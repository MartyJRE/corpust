import type { CorpusMeta, KwicResult, QueryLayer } from "@/types";
import { formatDuration } from "@/lib/utils";

export interface StatusBarProps {
  corpus: CorpusMeta | null;
  result: KwicResult | null;
  layer: QueryLayer;
}

export function StatusBar({ corpus, result, layer }: StatusBarProps) {
  return (
    <div className="cx-statusbar">
      <div className="cx-statusbar-left">
        {corpus ? (
          <>
            <span className="cx-status-dim">corpus</span>
            <span>{corpus.name}</span>
            <span className="cx-sep">·</span>
            <span>{corpus.docCount.toLocaleString()} docs</span>
            <span className="cx-sep">·</span>
            <span>{corpus.tokenCount.toLocaleString()} tokens</span>
            {corpus.annotated && (
              <>
                <span className="cx-sep">·</span>
                <span style={{ color: "var(--accent)" }}>annotated</span>
              </>
            )}
          </>
        ) : (
          <span className="cx-status-dim">no corpus loaded</span>
        )}
      </div>
      <div className="cx-statusbar-right">
        {result && (
          <>
            <span>{result.total.toLocaleString()} hits</span>
            <span className="cx-sep">·</span>
            <span className={`cx-layer-chip cx-layer-${layer}`}>{layer}</span>
            <span className="cx-sep">·</span>
            <span className="cx-status-time">{formatDuration(result.elapsedMs)}</span>
          </>
        )}
      </div>
    </div>
  );
}
