import { Download, Eye, Search } from "lucide-react";
import { useEffect, useState } from "react";
import { DOCUMENTS } from "@/data";
import { type DocumentInfo, hasLiveData, listDocuments } from "@/lib/tauri";
import type { CorpusMeta } from "@/types";
import { basename, formatBuildTime, formatBytes, formatDate } from "@/lib/utils";

export interface CorpusDetailProps {
  corpus: CorpusMeta;
  onDismiss: () => void;
}

interface DocRow {
  file: string;
  title: string | null;
  author: string | null;
  year: number | null;
  tokens: number;
}

/** Render a value or a muted em-dash when it's absent. */
function orDash(value: string | number | null) {
  if (value === null || value === "") {
    return <span style={{ color: "var(--fg-subtle)" }}>—</span>;
  }
  return value;
}

export function CorpusDetail({ corpus, onDismiss }: CorpusDetailProps) {
  const wps = Math.round(corpus.tokenCount / (corpus.buildMs / 1000));

  const [docs, setDocs] = useState<DocumentInfo[] | null>(null);
  useEffect(() => {
    setDocs(null);
    if (!hasLiveData(corpus.id)) return;
    let cancelled = false;
    listDocuments(corpus.id)
      .then((d) => {
        if (!cancelled) setDocs(d);
      })
      .catch((e) => console.error("listDocuments failed:", e));
    return () => {
      cancelled = true;
    };
  }, [corpus.id]);

  // Real corpora carry title / author / year extracted from the document
  // body at index time (any of which may be absent). Fixtures supply their
  // own demo metadata. A muted em-dash stands in for a missing field.
  const docRows: DocRow[] = docs
    ? docs.map((d) => ({
        file: basename(d.path),
        title: d.title ?? null,
        author: d.author ?? null,
        year: d.year ?? null,
        tokens: d.tokenCount,
      }))
    : DOCUMENTS.map((d) => ({
        file: d.id,
        title: d.title ?? null,
        author: d.author ?? null,
        year: d.year ?? null,
        tokens: d.tokens,
      }));
  return (
    <div className="cx-detail">
      <div className="cx-detail-head">
        <div>
          <div className="cx-detail-sub">corpus · {corpus.kind}</div>
          <h1 className="cx-detail-title">{corpus.name}</h1>
          <div className="cx-detail-sub">{corpus.indexPath}</div>
        </div>
        <div className="cx-detail-actions">
          <button type="button" className="cx-btn cx-btn-outline">
            <Eye size={13} />
            reveal
          </button>
          <button type="button" className="cx-btn cx-btn-outline">
            <Download size={13} />
            export
          </button>
          <button type="button" className="cx-btn cx-btn-primary" onClick={onDismiss}>
            <Search size={13} />
            query
          </button>
        </div>
      </div>

      <div className="cx-stat-row">
        <div className="cx-stat">
          <div className="cx-stat-label">documents</div>
          <div className="cx-stat-val">{corpus.docCount.toLocaleString()}</div>
        </div>
        <div className="cx-stat">
          <div className="cx-stat-label">tokens</div>
          <div className="cx-stat-val">{corpus.tokenCount.toLocaleString()}</div>
        </div>
        <div className="cx-stat">
          <div className="cx-stat-label">types</div>
          <div className="cx-stat-val">{corpus.types.toLocaleString()}</div>
        </div>
        <div className="cx-stat">
          <div className="cx-stat-label">index size</div>
          <div className="cx-stat-val">{formatBytes(corpus.sizeOnDisk)}</div>
        </div>
      </div>

      <div className="cx-section-h">
        <span>metadata</span>
      </div>
      <dl className="cx-meta-grid">
        <dt>source</dt>
        <dd>{corpus.sourcePath}</dd>
        <dt>index</dt>
        <dd>{corpus.indexPath}</dd>
        <dt>languages</dt>
        <dd>{corpus.languages.join(", ")}</dd>
        <dt>annotated</dt>
        <dd>
          {corpus.annotated ? (
            <span style={{ color: "var(--accent)" }}>yes · TreeTagger</span>
          ) : (
            "no"
          )}
        </dd>
        <dt>tokeniser</dt>
        <dd>{corpus.tokeniser}</dd>
        <dt>annotator</dt>
        <dd>{corpus.annotator ?? <span style={{ color: "var(--fg-subtle)" }}>—</span>}</dd>
        <dt>avg doc length</dt>
        <dd>{corpus.avgDocLen.toLocaleString()} tokens</dd>
        <dt>built at</dt>
        <dd>
          {formatDate(corpus.builtAt)}{" "}
          <span style={{ color: "var(--fg-subtle)" }}>
            · {formatBuildTime(corpus.buildMs)} ({wps.toLocaleString()} wps)
          </span>
        </dd>
      </dl>

      <div className="cx-section-h">
        <span>documents</span>
        <span className="sub">{docRows.length.toLocaleString()} files</span>
      </div>
      <table className="cx-doc-table">
        <thead>
          <tr>
            <th>file</th>
            <th>title</th>
            <th>author</th>
            <th className="num">year</th>
            <th className="num">tokens</th>
          </tr>
        </thead>
        <tbody>
          {docRows.map((d) => (
            <tr key={d.file}>
              <td style={{ color: "var(--fg-muted)" }}>{d.file}</td>
              <td>{orDash(d.title)}</td>
              <td>{orDash(d.author)}</td>
              <td className="num">{orDash(d.year)}</td>
              <td className="num">{d.tokens.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
