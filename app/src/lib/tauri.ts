// Thin typed wrapper over Tauri's `invoke`. Today these are stubs — real
// implementations land once the commands in app/src-tauri/src/commands.rs
// are fleshed out. UI code reaches through these fns so swapping fixture
// data for real IPC is a single-file change.

import type {
  BuildProgress,
  BuildRequest,
  Collocate,
  CorpusMeta,
  KwicRequest,
  KwicResult,
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
}

// Lazy import so calls outside a Tauri runtime (e.g. Storybook) don't crash.
async function invokeSafe<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(command, args);
}

export async function listCorpora(): Promise<CorpusMeta[]> {
  return invokeSafe<CorpusMeta[]>("list_corpora");
}

export async function openCorpus(indexPath: string): Promise<CorpusMeta> {
  return invokeSafe<CorpusMeta>("open_corpus", { indexPath });
}

export async function runKwic(req: KwicRequest): Promise<KwicResult> {
  return invokeSafe<KwicResult>("run_kwic", { req });
}

export async function runCollocates(req: CollocatesRequest): Promise<CollocatesResult> {
  return invokeSafe<CollocatesResult>("run_collocates", { req });
}

export async function buildIndex(req: BuildRequest): Promise<CorpusMeta> {
  return invokeSafe<CorpusMeta>("build_index", { req });
}

/** True when running inside the Tauri shell. Frontend can fall back to
 *  fixture data when false (Storybook, vite-only dev). */
export function inTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export type { BuildProgress };
