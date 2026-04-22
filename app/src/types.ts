// Domain types exposed to the frontend. Mirror the Rust-side
// corpust-core / corpust-index / corpust-query.

export type CorpusKind = "literary" | "legal" | "news" | "mixed";

export interface CorpusMeta {
  id: string;
  name: string;
  kind: CorpusKind;
  indexPath: string;
  sourcePath: string;
  annotated: boolean;
  docCount: number;
  tokenCount: number;
  types: number;
  avgDocLen: number;
  builtAt: string;
  buildMs: number;
  languages: string[];
  tokeniser: string;
  annotator: string | null;
  sizeOnDisk: number;
}

export type QueryLayer = "word" | "lemma" | "pos";

export interface KwicRequest {
  corpusId: string;
  term: string;
  layer: QueryLayer;
  context: number;
  limit: number;
}

export interface KwicHit {
  docId: string;
  pos: number;
  left: string;
  hit: string;
  right: string;
  lemma?: string;
  pos_tag?: string;
}

export interface KwicResult {
  hits: KwicHit[];
  elapsedMs: number;
  truncated: boolean;
}

export interface ExpandedHit {
  docTitle: string;
  docMeta: string;
  before: string;
  match: string;
  after: string;
}

export interface Collocate {
  word: string;
  pos: string;
  leftCount: number;
  rightCount: number;
  total: number;
  logDice: number;
  mi: number;
  z: number;
  dist: number;
}

export interface FreqRow {
  word?: string;
  tag?: string;
  label?: string;
  count: number;
  pct: number;
}

export interface DocFreqRow {
  doc: string;
  hits: number;
  per1m: number;
}

export interface DocumentMeta {
  id: string;
  title: string;
  author: string;
  year: number;
  tokens: number;
}

export interface RecentQuery {
  id: number;
  layer: QueryLayer;
  term: string;
  hits: number;
  corpus: string;
}

export interface BuildRequest {
  sourcePath: string;
  outPath: string;
  annotate: boolean;
}

export interface BuildProgress {
  taskId: string;
  phase: "idle" | "reading" | "indexing" | "annotating" | "committing" | "done" | "failed";
  docsSeen: number;
  docsTotal: number | null;
  elapsedMs: number;
  error?: string;
}

export type MainView = "search" | "corpus" | "settings";
export type SubView = "kwic" | "coll" | "freq";
export type SortMode = "left1" | "right1" | "doc";
export type CollMetric = "logDice" | "mi" | "z";
export type FreqBy = "word" | "pos";
