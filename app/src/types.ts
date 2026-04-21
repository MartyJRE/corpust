// Domain types exposed to the frontend. Keep these in sync with the
// Rust side — they mirror corpust-core + corpust-index + corpust-query.

export interface CorpusMeta {
  /** Stable name used in the UI + in URLs/paths. */
  id: string;
  /** Human-readable label. */
  name: string;
  /** Filesystem path to the on-disk Tantivy index. */
  indexPath: string;
  /** Filesystem path of the source directory the corpus was built from. */
  sourcePath: string;
  /** Whether the corpus was built with TreeTagger annotation (lemma + POS). */
  annotated: boolean;
  /** Total documents. */
  docCount: number;
  /** Approximate total token count (may be an estimate for unannotated builds). */
  tokenCount: number;
  /** ISO-8601 timestamp of index build completion. */
  builtAt: string;
  /** Tagger provenance, if annotated — e.g. "treetagger-english". */
  taggerId?: string;
}

export type QueryLayer = "word" | "lemma" | "pos";

export interface KwicRequest {
  corpusId: string;
  term: string;
  layer: QueryLayer;
  /** Tokens of context on each side. */
  context: number;
  /** Max hits to return. */
  limit: number;
}

export interface KwicHit {
  /** Document index inside the corpus. */
  docId: number;
  /** Relative path of the source file — shown as the left-most column. */
  path: string;
  /** Left context, N tokens. */
  left: string;
  /** The hit token (the thing that matched the query). */
  hit: string;
  /** Right context, N tokens. */
  right: string;
}

export interface KwicResult {
  hits: KwicHit[];
  /** Query wall-clock duration, rendered in the status bar. */
  elapsedMs: number;
  /** Whether the `limit` was reached (there might be more hits). */
  truncated: boolean;
}

export interface BuildRequest {
  /** Directory containing .txt files. */
  sourcePath: string;
  /** Where to write the index. */
  outPath: string;
  /** Run TreeTagger during indexing to populate lemma + POS. */
  annotate: boolean;
}

export interface BuildProgress {
  taskId: string;
  /** 0..1 */
  phase: "reading" | "indexing" | "committing" | "done" | "failed";
  docsSeen: number;
  docsTotal: number | null;
  elapsedMs: number;
  error?: string;
}
