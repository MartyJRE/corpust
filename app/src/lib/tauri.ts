// Thin typed wrapper over Tauri's `invoke`. Everything the React side
// does against the Rust backend goes through these functions — the UI
// never talks to `@tauri-apps/api/core` directly.

import { invoke } from "@tauri-apps/api/core";
import type {
  BuildProgress,
  BuildRequest,
  CorpusMeta,
  KwicRequest,
  KwicResult,
} from "@/types";

export async function listCorpora(): Promise<CorpusMeta[]> {
  return invoke<CorpusMeta[]>("list_corpora");
}

export async function openCorpus(indexPath: string): Promise<CorpusMeta> {
  return invoke<CorpusMeta>("open_corpus", { indexPath });
}

export async function runKwic(req: KwicRequest): Promise<KwicResult> {
  return invoke<KwicResult>("run_kwic", { req });
}

export async function buildIndex(req: BuildRequest): Promise<string> {
  // Returns a task id; progress events stream via the `build:progress`
  // event channel.
  return invoke<string>("build_index", { req });
}

export type { BuildProgress };
