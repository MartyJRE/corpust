//! Tauri command handlers. Frontend talks to these via `invoke`.
//!
//! Real implementations land here — they drive `corpust-io`,
//! `corpust-index`, `corpust-query`, and `corpust-tagger` directly and
//! keep a process-local registry of opened corpora under the
//! returned `corpusId` / `taskId` handles.

use crate::{
    AppState, BuildRequest, Collocate as CollocateDto, CollocatesRequest, CollocatesResult,
    CorpusMeta, CorpusMetaEnvelope, KwicHit as KwicHitDto, KwicRequest, KwicResult, OpenedCorpus,
    TaggerKind,
};
use corpust_annotate::{Annotator, treetagger::TreeTagger};
use corpust_index::{CorpusIndex, DEFAULT_CONTEXT};
use corpust_io::paths;
use corpust_query::{KwicRequest as CoreKwicRequest, kwic as run_core_kwic};
use corpust_tagger::Tagger as RustTagger;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager, State};

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

/// Scan the platform data directory and return every persisted corpus.
///
/// Disk is the source of truth — corpora survive restarts because the
/// build step writes `<slug>/metadata.json` next to the index. The
/// in-memory `AppState.corpora` registry is only a cache of opened
/// handles; we fall back to disk for everything else.
#[tauri::command]
pub fn list_corpora() -> Result<Vec<CorpusMeta>, String> {
    let root = match paths::corpora_root() {
        Ok(p) => p,
        Err(e) => return Err(format!("resolving data dir: {e:#}")),
    };
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let entries = std::fs::read_dir(&root)
        .map_err(|e| format!("reading {}: {e}", root.display()))?;
    for entry in entries.filter_map(Result::ok) {
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let meta_file = entry.path().join("metadata.json");
        if !meta_file.exists() {
            continue;
        }
        match read_metadata_file(&meta_file) {
            Ok(meta) => out.push(meta),
            Err(e) => eprintln!("skipping {}: {e:#}", meta_file.display()),
        }
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    Ok(out)
}

/// Open a corpus by slug (the `id` field returned from `list_corpora`
/// or `build_index`). Registers the handle in `AppState` so subsequent
/// KWIC / collocate calls hit the same instance.
#[tauri::command]
pub fn open_corpus(
    state: State<'_, AppState>,
    id: String,
) -> Result<CorpusMeta, String> {
    let (index, meta) = load_from_disk(&id)?;
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
pub fn run_collocates(
    state: State<'_, AppState>,
    req: CollocatesRequest,
) -> Result<CollocatesResult, String> {
    // Pull a large KWIC result with enough context to cover both
    // window sides, then aggregate collocates. Cap hits at 5000 —
    // enough for meaningful collocations on common terms without
    // blowing up on "the".
    const HIT_CAP: usize = 5000;
    let lw = req.left_window.min(30);
    let rw = req.right_window.min(30);
    if lw == 0 && rw == 0 {
        return Err("collocation window must include at least one side".to_owned());
    }
    let context = lw.max(rw).max(1);
    let kreq = CoreKwicRequest::new(&req.term)
        .layer(req.layer.into())
        .context(context)
        .limit(HIT_CAP);
    let t0 = Instant::now();
    let hits = with_corpus(&state, &req.corpus_id, |index| {
        run_core_kwic(index, kreq).map_err(|e| format!("kwic failed: {e:#}"))
    })?;

    // Count word occurrences per side, honoring asymmetric L/R
    // windows. The KWIC call fetched `context` tokens of each side,
    // so we now trim each side's stream to the requested L or R
    // window (left: last N tokens; right: first N tokens).
    use std::collections::HashMap;
    let mut left_counts: HashMap<String, u32> = HashMap::new();
    let mut right_counts: HashMap<String, u32> = HashMap::new();
    let mut window_tokens: u32 = 0;

    for h in &hits {
        if lw > 0 {
            let left_toks: Vec<String> = tokenize_for_collocates(&h.left).collect();
            let start = left_toks.len().saturating_sub(lw);
            for w in &left_toks[start..] {
                *left_counts.entry(w.clone()).or_default() += 1;
                window_tokens += 1;
            }
        }
        if rw > 0 {
            for w in tokenize_for_collocates(&h.right).take(rw) {
                *right_counts.entry(w).or_default() += 1;
                window_tokens += 1;
            }
        }
    }

    // Merge sides + rank by total.
    let mut merged: HashMap<String, (u32, u32)> = HashMap::new();
    for (w, l) in left_counts {
        merged.entry(w).or_default().0 = l;
    }
    for (w, r) in right_counts {
        merged.entry(w).or_default().1 = r;
    }
    let mut vec: Vec<(String, u32, u32)> = merged
        .into_iter()
        .map(|(w, (l, r))| (w, l, r))
        .collect();
    vec.sort_by(|a, b| (b.1 + b.2).cmp(&(a.1 + a.2)));
    vec.truncate(req.limit.max(1).min(200));

    let collocates: Vec<CollocateDto> = vec
        .into_iter()
        .map(|(w, l, r)| {
            let total = l + r;
            // Placeholder stats until we wire corpus-wide term
            // frequencies. log2(total+1) gives a monotonic proxy that
            // spreads the scatter's y-axis readably.
            let score = ((total + 1) as f64).log2();
            CollocateDto {
                word: w,
                pos: String::new(),
                left_count: l,
                right_count: r,
                total,
                log_dice: score,
                mi: score,
                z: score,
                dist: 0,
            }
        })
        .collect();

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok(CollocatesResult {
        collocates,
        elapsed_ms,
        node_hits: hits.len() as u32,
        window_tokens,
    })
}

/// Split a KWIC context string into collocate candidates.
/// - whitespace-separated,
/// - lowercased,
/// - punctuation-trimmed at ends,
/// - filter empty/single-char tokens,
/// - filter pure-digit tokens.
fn tokenize_for_collocates(s: &str) -> impl Iterator<Item = String> + '_ {
    s.split_whitespace().filter_map(|tok| {
        let trimmed: String = tok
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        if trimmed.len() < 2 {
            return None;
        }
        if trimmed.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        Some(trimmed)
    })
}

#[tauri::command]
pub fn run_kwic(
    state: State<'_, AppState>,
    req: KwicRequest,
) -> Result<KwicResult, String> {
    let context = if req.context == 0 { DEFAULT_CONTEXT } else { req.context };
    let limit = req.limit.max(1);
    let kreq = CoreKwicRequest::new(&req.term)
        .layer(req.layer.into())
        .context(context)
        .limit(limit);

    let t0 = Instant::now();
    let hits = with_corpus(&state, &req.corpus_id, |index| {
        run_core_kwic(index, kreq).map_err(|e| format!("kwic failed: {e:#}"))
    })?;
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

/// Runs the build on a worker thread so the UI event loop stays
/// responsive. Tauri routes sync `fn` commands onto the main thread
/// by default — long-running work in that context freezes the
/// window. Making the command `async fn` + punting to
/// `spawn_blocking` keeps both sides happy: no UI freeze, and the
/// file/I-O/annotator code inside stays synchronous.
#[tauri::command]
pub async fn build_index(
    app: AppHandle,
    req: BuildRequest,
) -> Result<CorpusMeta, String> {
    let started = Instant::now();
    let handle = tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let result = build_index_inner(&app, &state, &req, started);
        if let Err(ref msg) = result {
            emit_failure(&app, started, msg);
        }
        result
    });
    handle
        .await
        .map_err(|e| format!("build task failed to join: {e}"))?
}

fn build_index_inner(
    app: &AppHandle,
    state: &State<'_, AppState>,
    req: &BuildRequest,
    started: Instant,
) -> Result<CorpusMeta, String> {
    let source_path = PathBuf::from(&req.source_path);

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

    // Resolve the display name and derive an on-disk slug. Collisions
    // get `-2`, `-3`, … appended so a user can re-build against the
    // same folder without overwriting the previous index.
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
    let base_slug = paths::slugify(&name);
    let slug = paths::unique_slug(&base_slug)
        .map_err(|e| format!("allocating slug for {name:?}: {e:#}"))?;
    let corpus_dir = paths::corpus_dir(&slug)
        .map_err(|e| format!("resolving corpus dir: {e:#}"))?;
    let out_path = corpus_dir.join("index");
    std::fs::create_dir_all(&corpus_dir)
        .map_err(|e| format!("creating {}: {e}", corpus_dir.display()))?;

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
        match req.tagger {
            TaggerKind::Rust => {
                let (par, abbr_path) = resolve_treetagger_bundle(app, "english")?;
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
                let tg = RustTagger::load(&par, "english", abbr).map_err(|e| {
                    format!("loading pure-Rust tagger from {}: {e:#}", par.display())
                })?;
                let id = tg.id().to_owned();
                (Some(Box::new(tg) as Box<dyn Annotator + Sync>), Some(id))
            }
            TaggerKind::Subprocess => {
                let bundle_root = resolve_treetagger_bundle_root(app)?;
                let tg = TreeTagger::from_bundle(&bundle_root, "english").map_err(|e| {
                    format!(
                        "loading subprocess tagger from {}: {e:#}",
                        bundle_root.display()
                    )
                })?;
                let id = tg.id().to_owned();
                (Some(Box::new(tg) as Box<dyn Annotator + Sync>), Some(id))
            }
        }
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

    let mut meta = CorpusMeta::stub(
        slug.clone(),
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

    // Persist the metadata sidecar so this corpus shows up on next
    // `list_corpora` call. Done before we mutate the registry —
    // failing here means the index is orphaned but the state stays
    // clean, and the user can retry.
    write_metadata_file(&corpus_dir.join("metadata.json"), &meta)
        .map_err(|e| format!("writing metadata: {e:#}"))?;

    emit_progress(app, started, "done", doc_count as u64, Some(doc_count as u64));
    state
        .corpora
        .lock()
        .expect("corpus registry poisoned")
        .insert(
            slug,
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

/// Run `f` against a corpus handle, lazy-opening it from disk if it
/// isn't already cached in the registry. The handle stays in the
/// registry afterwards so the next query is cheap.
fn with_corpus<F, R>(state: &State<'_, AppState>, id: &str, f: F) -> Result<R, String>
where
    F: FnOnce(&CorpusIndex) -> Result<R, String>,
{
    {
        let reg = state.corpora.lock().expect("corpus registry poisoned");
        if let Some(c) = reg.get(id) {
            return f(&c.index);
        }
    }
    let (index, meta) = load_from_disk(id)?;
    let result = f(&index);
    state
        .corpora
        .lock()
        .expect("corpus registry poisoned")
        .insert(id.to_owned(), OpenedCorpus { index, meta });
    result
}

/// Open an existing corpus from disk by slug. Returns the tantivy
/// handle plus the persisted metadata.
fn load_from_disk(slug: &str) -> Result<(CorpusIndex, CorpusMeta), String> {
    let corpus_dir = paths::corpus_dir(slug)
        .map_err(|e| format!("resolving corpus dir for {slug}: {e:#}"))?;
    let meta_file = corpus_dir.join("metadata.json");
    if !meta_file.exists() {
        return Err(format!(
            "no corpus named {slug:?} (expected {})",
            meta_file.display()
        ));
    }
    let meta = read_metadata_file(&meta_file)
        .map_err(|e| format!("reading {}: {e:#}", meta_file.display()))?;
    let index_dir = corpus_dir.join("index");
    let index = CorpusIndex::open(&index_dir)
        .map_err(|e| format!("opening {}: {e:#}", index_dir.display()))?;
    Ok((index, meta))
}

fn read_metadata_file(path: &Path) -> anyhow::Result<CorpusMeta> {
    let bytes = std::fs::read(path)?;
    let envelope: CorpusMetaEnvelope = serde_json::from_slice(&bytes)?;
    // Future schema bumps get their migrations here.
    if envelope.schema_version != CorpusMetaEnvelope::CURRENT_VERSION {
        anyhow::bail!(
            "unsupported metadata schema version {} (expected {})",
            envelope.schema_version,
            CorpusMetaEnvelope::CURRENT_VERSION
        );
    }
    Ok(envelope.corpus)
}

fn write_metadata_file(path: &Path, meta: &CorpusMeta) -> anyhow::Result<()> {
    let envelope = CorpusMetaEnvelope::wrap(meta.clone());
    let json = serde_json::to_vec_pretty(&envelope)?;
    std::fs::write(path, json)?;
    Ok(())
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
fn resolve_treetagger_bundle(
    app: &AppHandle,
    language: &str,
) -> Result<(PathBuf, PathBuf), String> {
    let bundle = resolve_treetagger_bundle_root(app)?;
    let par = bundle.join("lib").join(format!("{language}.par"));
    if !par.exists() {
        return Err(format!(
            "TreeTagger bundle {} has no {}.par under lib/",
            bundle.display(),
            language
        ));
    }
    let abbr = bundle
        .join("lib")
        .join(format!("{language}-abbreviations"));
    Ok((par, abbr))
}

/// Locate the TreeTagger bundle across dev and packaged modes.
///
/// Search order:
///   1. `$CORPUST_TREETAGGER_BUNDLE` — explicit override.
///   2. Tauri's resource directory (`.app/Contents/Resources/` on
///      macOS), including the `_up_`-mangled path the bundler
///      generates for resources declared with `..` in tauri.conf.json.
///   3. Directories adjacent to or above the running binary
///      (`target/debug/corpust-ui` → up to the repo root).
///   4. Paths relative to the process cwd (works for `cargo run`
///      from the repo root).
fn resolve_treetagger_bundle_root(app: &AppHandle) -> Result<PathBuf, String> {
    use tauri::Manager;

    let mut tried: Vec<PathBuf> = Vec::new();
    let try_path = |p: PathBuf, tried: &mut Vec<PathBuf>| -> Option<PathBuf> {
        let has_lib = p.join("lib").exists();
        tried.push(p.clone());
        has_lib.then_some(p)
    };

    // 1. Env var
    if let Ok(v) = std::env::var("CORPUST_TREETAGGER_BUNDLE") {
        let candidate = PathBuf::from(v);
        if let Some(found) = try_path(candidate, &mut tried) {
            return Ok(found);
        }
    }

    // 2. Tauri resource dir (packaged .app). The bundler rewrites
    // `../../resources/treetagger` in tauri.conf.json to
    // `_up_/_up_/resources/treetagger` under Contents/Resources.
    if let Ok(resource_root) = app.path().resource_dir() {
        for sub in [
            "resources/treetagger",
            "_up_/_up_/resources/treetagger",
            "_up_/resources/treetagger",
        ] {
            let candidate = resource_root.join(sub);
            if let Some(found) = try_path(candidate, &mut tried) {
                return Ok(found);
            }
        }
    }

    // 3. Relative to the running binary. On macOS `.app`s the
    // layout is `<app>.app/Contents/MacOS/<bin>` and resources live
    // at `<app>.app/Contents/Resources/`; for dev builds the binary
    // sits at `target/{debug,release}/<bin>` and the repo's
    // `resources/treetagger/` is a few levels up.
    if let Ok(exe) = std::env::current_exe() {
        let mut cursor = exe.parent().map(|p| p.to_path_buf());
        for _ in 0..8 {
            let Some(dir) = cursor.clone() else { break };
            for sub in [
                "resources/treetagger",
                "../Resources/resources/treetagger",
                "../Resources/_up_/_up_/resources/treetagger",
            ] {
                let candidate = dir.join(sub);
                if let Some(found) = try_path(candidate, &mut tried) {
                    return Ok(found);
                }
            }
            cursor = dir.parent().map(|p| p.to_path_buf());
        }
    }

    // 4. cwd-relative — last-chance fallback, useful for
    // `cargo run` from the repo root.
    for rel in ["resources/treetagger", "../resources/treetagger", "../../resources/treetagger"] {
        if let Some(found) = try_path(PathBuf::from(rel), &mut tried) {
            return Ok(found);
        }
    }

    let tried_list = tried
        .iter()
        .map(|p| format!("  - {}", p.display()))
        .collect::<Vec<_>>()
        .join("\n");
    Err(format!(
        "no TreeTagger bundle found. Tried:\n{tried_list}\n\nSet \
         CORPUST_TREETAGGER_BUNDLE to point at the bundle root \
         explicitly (e.g. export \
         CORPUST_TREETAGGER_BUNDLE=/path/to/resources/treetagger)."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_file(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "corpust-meta-{}-{}.json",
            name,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }

    #[test]
    fn metadata_round_trips_through_envelope() {
        let path = tmp_file("round-trip");
        let mut meta = CorpusMeta::stub(
            "my-corpus".to_owned(),
            "My Corpus".to_owned(),
            "/tmp/fake/index".to_owned(),
        );
        meta.doc_count = 42;
        meta.token_count = 1234;
        meta.annotated = true;
        meta.annotator = Some("treetagger-rs-english".to_owned());

        write_metadata_file(&path, &meta).unwrap();
        let read_back = read_metadata_file(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(read_back.id, "my-corpus");
        assert_eq!(read_back.name, "My Corpus");
        assert_eq!(read_back.doc_count, 42);
        assert_eq!(read_back.token_count, 1234);
        assert!(read_back.annotated);
        assert_eq!(read_back.annotator.as_deref(), Some("treetagger-rs-english"));
    }

    #[test]
    fn metadata_read_rejects_future_schema_versions() {
        let path = tmp_file("bad-schema");
        let bogus = serde_json::json!({
            "schemaVersion": 999,
            "corpus": CorpusMeta::stub("x".into(), "x".into(), "/x".into()),
        });
        std::fs::write(&path, serde_json::to_vec(&bogus).unwrap()).unwrap();
        let err = read_metadata_file(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(
            err.to_string().contains("unsupported metadata schema version"),
            "unexpected error: {err}"
        );
    }
}
