// Thin typed wrapper over Tauri's `invoke`. Today these are stubs — real
// implementations land once the commands in app/src-tauri/src/commands.rs
// are fleshed out. UI code reaches through these fns so swapping fixture
// data for real IPC is a single-file change.

import { CORPORA } from "@/data";
import type {
  BuildProgress,
  BuildRequest,
  Collocate,
  CorpusMeta,
  KwicRequest,
  QueryLayer,
} from "@/types";

export interface CollocatesRequest {
  corpusId: string;
  term: string;
  layer: QueryLayer;
  /** Tokens to consider on the left of the node. 0 = skip left. */
  leftWindow: number;
  /** Tokens to consider on the right of the node. 0 = skip right. */
  rightWindow: number;
  limit: number;
}

export interface CollocatesResult {
  collocates: Collocate[];
  elapsedMs: number;
  nodeHits: number;
  windowTokens: number;
  /** True when the full-corpus scan hit its safety ceiling, so scores
   *  reflect a large sample rather than every node occurrence. */
  truncated: boolean;
}

/** Raw KWIC hit as returned by the backend — distinct from the UI's
 *  `KwicHit` (numeric doc id, carries `path` + `hitPosition`). The App
 *  adapts these into the UI shape. */
export interface KwicHitRaw {
  docId: number;
  path: string;
  hitPosition: number;
  left: string;
  hit: string;
  right: string;
}

export interface KwicResultRaw {
  hits: KwicHitRaw[];
  elapsedMs: number;
  truncated: boolean;
}

export interface DocumentInfo {
  docId: number;
  path: string;
  tokenCount: number;
  /** Title / author / year extracted from the document body at index
   *  time. Null when the backend extractor found nothing confidently. */
  title: string | null;
  author: string | null;
  year: number | null;
}

export interface FrequenciesRequest {
  corpusId: string;
  layer: QueryLayer;
  limit: number;
}

export interface FreqResultRow {
  term: string;
  count: number;
  pct: number;
}

export interface FrequenciesResult {
  rows: FreqResultRow[];
  totalTokens: number;
  elapsedMs: number;
}

export interface TermDistRequest {
  corpusId: string;
  term: string;
  layer: QueryLayer;
  buckets: number;
}

export interface DocTermCount {
  docId: number;
  path: string;
  hits: number;
  tokenCount: number;
}

export interface TermDistResult {
  docCounts: DocTermCount[];
  /** Per-bucket occurrence counts over the corpus position axis. */
  dispersion: number[];
  totalHits: number;
  elapsedMs: number;
}

export interface CollocateDistanceRequest {
  corpusId: string;
  term: string;
  layer: QueryLayer;
  leftWindow: number;
  rightWindow: number;
  limit: number;
}

export interface DistanceRow {
  word: string;
  total: number;
  /** One count per offset in `offsets` (same order). */
  counts: number[];
}

export interface CollocateDistanceResult {
  /** Signed slot offsets, e.g. [-3,-2,-1,1,2,3]. */
  offsets: number[];
  rows: DistanceRow[];
  nodeHits: number;
  truncated: boolean;
  elapsedMs: number;
}

export interface ExpandRequest {
  corpusId: string;
  docId: number;
  position: number;
  context: number;
}

export interface ExpandedContext {
  docId: number;
  path: string;
  before: string;
  hit: string;
  after: string;
  tokenCount: number;
}

// Lazy import so calls outside a Tauri runtime (e.g. Storybook) don't crash.
async function invokeSafe<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(command, args);
}

export async function listCorpora(): Promise<CorpusMeta[]> {
  return invokeSafe<CorpusMeta[]>("list_corpora");
}

export async function runKwic(req: KwicRequest): Promise<KwicResultRaw> {
  return invokeSafe<KwicResultRaw>("run_kwic", { req });
}

export async function runCollocates(req: CollocatesRequest): Promise<CollocatesResult> {
  return invokeSafe<CollocatesResult>("run_collocates", { req });
}

export async function buildIndex(req: BuildRequest): Promise<CorpusMeta> {
  return invokeSafe<CorpusMeta>("build_index", { req });
}

export async function listDocuments(corpusId: string): Promise<DocumentInfo[]> {
  return invokeSafe<DocumentInfo[]>("list_documents", { corpusId });
}

export async function runFrequencies(req: FrequenciesRequest): Promise<FrequenciesResult> {
  return invokeSafe<FrequenciesResult>("run_frequencies", { req });
}

export async function runCollocateDistance(
  req: CollocateDistanceRequest,
): Promise<CollocateDistanceResult> {
  return invokeSafe<CollocateDistanceResult>("run_collocate_distance", { req });
}

export async function runTermDistribution(req: TermDistRequest): Promise<TermDistResult> {
  return invokeSafe<TermDistResult>("run_term_distribution", { req });
}

export async function expandContext(req: ExpandRequest): Promise<ExpandedContext> {
  return invokeSafe<ExpandedContext>("expand_context", { req });
}

/** True when running inside the Tauri shell. Frontend can fall back to
 *  fixture data when false (Storybook, vite-only dev). */
export function inTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

/** A corpus id that belongs to the baked-in demo fixtures (`@/data`)
 *  rather than a real, backend-registered corpus. Views use this to
 *  decide whether to hit IPC or render fixture data. */
export function isFixtureCorpus(id: string): boolean {
  return CORPORA.some((c) => c.id === id);
}

/** Live data is available for a corpus only inside Tauri and only when
 *  it isn't one of the demo fixtures. */
export function hasLiveData(id: string): boolean {
  return inTauri() && !isFixtureCorpus(id);
}

export type { BuildProgress };
