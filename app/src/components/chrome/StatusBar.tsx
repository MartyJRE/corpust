import type { CorpusMeta, KwicResult, QueryLayer } from "@/types";
import { formatDuration } from "@/lib/utils";

export interface StatusBarProps {
  corpus: CorpusMeta | null;
  result: KwicResult | null;
  layer: QueryLayer;
  memory?: number;
}

export function StatusBar({ corpus, result, layer, memory = 0.42 }: StatusBarProps) {
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
        <span className="cx-status-mem" title="memory">
          mem
          <span className="cx-status-mem-bar">
            <span style={{ display: "block", height: "100%", width: `${memory * 100}%`, background: "var(--fg-subtle)" }} />
          </span>
          {Math.round(memory * 100)}%
        </span>
        {result && (
          <>
            <span className="cx-sep">·</span>
            <span>{result.hits.length.toLocaleString()} hits</span>
            <span className="cx-sep">·</span>
            <span className={`cx-layer-chip cx-layer-${layer}`}>{layer}</span>
            <span className="cx-sep">·</span>
            <span className="cx-status-time">{formatDuration(result.elapsedMs)}</span>
            {result.truncated && (
              <>
                <span className="cx-sep">·</span>
                <span className="cx-warn">truncated</span>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
