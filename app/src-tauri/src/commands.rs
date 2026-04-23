//! Tauri command handlers. Frontend talks to these via `invoke`.
//!
//! Real implementations land here — they drive `corpust-io`,
//! `corpust-index`, `corpust-query`, and `corpust-tagger` directly and
//! keep a process-local registry of opened corpora under the
//! returned `corpusId` / `taskId` handles.

use crate::{
    AppState, BuildRequest, CorpusMeta, KwicHit as KwicHitDto, KwicRequest, KwicResult,
    OpenedCorpus,
};
use corpust_annotate::Annotator;
use corpust_index::{CorpusIndex, DEFAULT_CONTEXT};
use corpust_query::{KwicRequest as CoreKwicRequest, kwic as run_core_kwic};
use corpust_tagger::Tagger as RustTagger;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, State};

const PROGRESS_EVENT: &str = "build:progress";

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct BuildProgress {
    phase: &'static str,
    docs_seen: u64,
    docs_total: Option<u64>,
    elapsed_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn emit_progress(
    app: &AppHandle,
    started: Instant,
    phase: &'static str,
    seen: u64,
    total: Option<u64>,
) {
    let _ = app.emit(
        PROGRESS_EVENT,
        BuildProgress {
            phase,
            docs_seen: seen,
            docs_total: total,
            elapsed_ms: started.elapsed().as_millis() as u64,
            error: None,
        },
    );
}

fn emit_failure(app: &AppHandle, started: Instant, message: &str) {
    let _ = app.emit(
        PROGRESS_EVENT,
        BuildProgress {
            phase: "failed",
            docs_seen: 0,
            docs_total: None,
            elapsed_ms: started.elapsed().as_millis() as u64,
            error: Some(message.to_owned()),
        },
    );
}

#[tauri::command]
pub fn list_corpora(state: State<'_, AppState>) -> Vec<CorpusMeta> {
    state
        .corpora
        .lock()
        .expect("corpus registry poisoned")
        .values()
        .map(|c| c.meta.clone())
        .collect()
}

#[tauri::command]
pub fn open_corpus(
    state: State<'_, AppState>,
    index_path: String,
) -> Result<CorpusMeta, String> {
    let path = PathBuf::from(&index_path);
    let index =
        CorpusIndex::open(&path).map_err(|e| format!("open {index_path}: {e:#}"))?;
    let id = fresh_id();
    let mut meta = CorpusMeta::stub(
        id.clone(),
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("corpus")
            .to_owned(),
        path.to_string_lossy().into_owned(),
    );
    meta.annotated = true; // assume until CorpusIndex exposes the flag
    meta.built_at = iso_now();
    meta.size_on_disk = dir_size(&path).unwrap_or(0);
    state
        .corpora
        .lock()
        .expect("corpus registry poisoned")
        .insert(
            id,
            OpenedCorpus {
                index,
                meta: meta.clone(),
            },
        );
    Ok(meta)
}

#[tauri::command]
pub fn run_kwic(
    state: State<'_, AppState>,
    req: KwicRequest,
) -> Result<KwicResult, String> {
    let reg = state.corpora.lock().expect("corpus registry poisoned");
    let opened = reg
        .get(&req.corpus_id)
        .ok_or_else(|| format!("no open corpus with id {}", req.corpus_id))?;
    let context = if req.context == 0 { DEFAULT_CONTEXT } else { req.context };
    let limit = req.limit.max(1);
    let kreq = CoreKwicRequest::new(&req.term)
        .layer(req.layer.into())
        .context(context)
        .limit(limit);

    let t0 = Instant::now();
    let hits = run_core_kwic(&opened.index, kreq)
        .map_err(|e| format!("kwic failed: {e:#}"))?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let truncated = hits.len() == limit;
    Ok(KwicResult {
        hits: hits
            .into_iter()
            .map(|h| KwicHitDto {
                doc_id: h.doc_id as u64,
                path: h.path.to_string_lossy().into_owned(),
                left: h.left,
                hit: h.hit,
                right: h.right,
            })
            .collect(),
        elapsed_ms,
        truncated,
    })
}

#[tauri::command]
pub fn build_index(
    app: AppHandle,
    state: State<'_, AppState>,
    req: BuildRequest,
) -> Result<CorpusMeta, String> {
    let started = Instant::now();
    let result = build_index_inner(&app, &state, &req, started);
    if let Err(ref msg) = result {
        emit_failure(&app, started, msg);
    }
    result
}

fn build_index_inner(
    app: &AppHandle,
    state: &State<'_, AppState>,
    req: &BuildRequest,
    started: Instant,
) -> Result<CorpusMeta, String> {
    let source_path = PathBuf::from(&req.source_path);
    let out_path = PathBuf::from(&req.out_path);

    if !source_path.exists() {
        return Err(format!(
            "source path {} does not exist (cwd is {})",
            source_path.display(),
            std::env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "<unknown>".to_owned())
        ));
    }
    if !source_path.is_dir() {
        return Err(format!("{} is not a directory", source_path.display()));
    }

    emit_progress(app, started, "reading", 0, None);
    let docs = corpust_io::read_text_dir(&source_path)
        .map_err(|e| format!("reading {}: {e:#}", source_path.display()))?;
    let doc_count = docs.len();
    let byte_count: usize = docs.iter().map(|d| d.text.len()).sum();
    if doc_count == 0 {
        return Err(format!(
            "no .txt files found under {} (cwd {})",
            source_path.display(),
            std::env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "<unknown>".to_owned())
        ));
    }

    let (tagger, tagger_id) = if req.annotate {
        let (par, abbr_path) = resolve_treetagger_bundle("english")?;
        let abbr = if abbr_path.exists() {
            std::fs::read_to_string(&abbr_path)
                .map_err(|e| format!("reading {}: {e}", abbr_path.display()))?
                .lines()
                .filter_map(|l| {
                    let t = l.trim();
                    (!t.is_empty() && !t.starts_with('#')).then(|| t.to_owned())
                })
                .collect()
        } else {
            Vec::new()
        };
        let tg = RustTagger::load(&par, "english", abbr)
            .map_err(|e| format!("loading tagger from {}: {e:#}", par.display()))?;
        let id = tg.id().to_owned();
        (Some(Box::new(tg) as Box<dyn Annotator + Sync>), Some(id))
    } else {
        (None, None)
    };

    let indexing_phase = if req.annotate { "annotating" } else { "indexing" };
    emit_progress(app, started, indexing_phase, 0, Some(doc_count as u64));

    let t_build = Instant::now();
    let index = CorpusIndex::create(&out_path)
        .map_err(|e| format!("creating index {}: {e:#}", out_path.display()))?;

    // Throttle event emission: the indexer fires the callback per
    // document. On fast workloads that's thousands of events per
    // second — emit only when the count meaningfully advances or
    // enough wall-clock has passed.
    let mut last_emitted = 0usize;
    let mut last_instant = Instant::now();
    index
        .add_documents_with_progress(docs, tagger.as_deref(), |seen| {
            let elapsed = last_instant.elapsed();
            if seen == doc_count
                || seen - last_emitted >= (doc_count / 200).max(1)
                || elapsed.as_millis() >= 100
            {
                emit_progress(app, started, indexing_phase, seen as u64, Some(doc_count as u64));
                last_emitted = seen;
                last_instant = Instant::now();
            }
        })
        .map_err(|e| format!("indexing failed: {e:#}"))?;
    let build_ms = t_build.elapsed().as_millis() as u64;
    emit_progress(app, started, "committing", doc_count as u64, Some(doc_count as u64));

    let id = fresh_id();
    let name = req
        .name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| {
            source_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("corpus")
                .to_owned()
        });
    let mut meta = CorpusMeta::stub(
        id.clone(),
        name,
        out_path.to_string_lossy().into_owned(),
    );
    meta.source_path = source_path.to_string_lossy().into_owned();
    meta.annotated = req.annotate;
    meta.doc_count = doc_count as u64;
    // Rough byte-based token approximation — a proper count needs
    // an aggregation pass over the index. Good enough for the UI's
    // "built: N tokens" header for now.
    meta.token_count = (byte_count / 6) as u64;
    meta.avg_doc_len = if doc_count > 0 { (byte_count / doc_count) as u64 } else { 0 };
    meta.built_at = iso_now();
    meta.build_ms = build_ms;
    meta.size_on_disk = dir_size(&out_path).unwrap_or(0);
    meta.annotator = tagger_id.clone();
    meta.tagger_id = tagger_id;
    emit_progress(app, started, "done", doc_count as u64, Some(doc_count as u64));
    state
        .corpora
        .lock()
        .expect("corpus registry poisoned")
        .insert(
            id,
            OpenedCorpus {
                index,
                meta: meta.clone(),
            },
        );
    Ok(meta)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fresh_id() -> String {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("corpus-{n}")
}

/// Recursive on-disk byte total for a directory. Silently returns
/// `None` if anything goes wrong — this is display-only and the UI
/// shouldn't fail the build over a stat error.
fn dir_size(path: &std::path::Path) -> Option<u64> {
    fn walk(path: &std::path::Path, total: &mut u64) -> std::io::Result<()> {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let md = entry.metadata()?;
            if md.is_dir() {
                walk(&entry.path(), total)?;
            } else {
                *total += md.len();
            }
        }
        Ok(())
    }
    let mut total = 0u64;
    walk(path, &mut total).ok()?;
    Some(total)
}

fn iso_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix:{secs}")
}

/// Locate the bundled TreeTagger parameter + abbreviations files.
///
/// The Tauri dev runtime cwd is usually `app/src-tauri/`, but packaged
/// apps can land elsewhere, so try a few common relative paths. Users
/// running a packaged build will eventually need a settings pane to
/// point us at the right location — tracked for the polish pass.
fn resolve_treetagger_bundle(language: &str) -> Result<(PathBuf, PathBuf), String> {
    let candidates = [
        PathBuf::from("resources/treetagger"),
        PathBuf::from("../resources/treetagger"),
        PathBuf::from("../../resources/treetagger"),
    ];
    for bundle in candidates {
        let par = bundle.join("lib").join(format!("{language}.par"));
        if par.exists() {
            let abbr = bundle
                .join("lib")
                .join(format!("{language}-abbreviations"));
            return Ok((par, abbr));
        }
    }
    Err(format!(
        "no TreeTagger bundle found; tried './resources/treetagger', \
         '../resources/treetagger', '../../resources/treetagger' from cwd {}",
        std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "<unknown>".to_owned())
    ))
}
