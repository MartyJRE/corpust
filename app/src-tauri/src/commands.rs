//! Tauri command handlers. Frontend talks to these via `invoke`.
//!
//! Real implementations land here — they drive `corpust-io`,
//! `corpust-index`, `corpust-query`, and `corpust-tagger` directly and
//! keep a process-local registry of opened corpora under the
//! returned `corpusId` / `taskId` handles.

use crate::{
    AppState, BuildRequest, Collocate as CollocateDto, CollocateDistanceRequest,
    CollocateDistanceResult, CollocatesRequest, CollocatesResult, CorpusMeta, CorpusMetaEnvelope,
    DistanceRow, DocTermCount as DocTermCountDto, DocumentInfo as DocumentInfoDto, ExpandRequest,
    ExpandedContext, FreqRow, FrequenciesRequest, FrequenciesResult, KwicHit as KwicHitDto,
    KwicRequest, KwicResult, OpenedCorpus, TaggerKind, TermDistRequest, TermDistResult,
};
use corpust_annotate::{Annotator, treetagger::TreeTagger};
use corpust_index::{CorpusIndex, DEFAULT_CONTEXT, QueryLayer};
use corpust_io::paths;
use corpust_query::{KwicRequest as CoreKwicRequest, kwic as run_core_kwic};
use corpust_tagger::Tagger as RustTagger;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;
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
/// Async so scanning the data directory + parsing each metadata sidecar
/// doesn't stall the UI thread at startup with many corpora present.
#[tauri::command]
pub async fn list_corpora() -> Result<Vec<CorpusMeta>, String> {
    tauri::async_runtime::spawn_blocking(list_corpora_inner)
        .await
        .map_err(|e| format!("list corpora task failed to join: {e}"))?
}

fn list_corpora_inner() -> Result<Vec<CorpusMeta>, String> {
    let root = match paths::corpora_root() {
        Ok(p) => p,
        Err(e) => return Err(format!("resolving data dir: {e:#}")),
    };
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let entries =
        std::fs::read_dir(&root).map_err(|e| format!("reading {}: {e}", root.display()))?;
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
/// Async so opening a large index from disk (Tantivy reader setup) runs
/// off the UI thread when the user switches corpora.
#[tauri::command]
pub async fn open_corpus(app: AppHandle, id: String) -> Result<CorpusMeta, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
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
    })
    .await
    .map_err(|e| format!("open corpus task failed to join: {e}"))?
}

/// Async so the full-corpus collocation scan runs on a worker thread —
/// Tauri routes sync `fn` commands onto the main thread, where a scan
/// over a frequent node ("the" in a 100M-token corpus) would freeze the
/// window. `spawn_blocking` keeps the index/I-O code synchronous while
/// the UI stays responsive (the frontend shows a loading state).
#[tauri::command]
pub async fn run_collocates(
    app: AppHandle,
    req: CollocatesRequest,
) -> Result<CollocatesResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        run_collocates_inner(&state, &req)
    })
    .await
    .map_err(|e| format!("collocates task failed to join: {e}"))?
}

fn run_collocates_inner(
    state: &State<'_, AppState>,
    req: &CollocatesRequest,
) -> Result<CollocatesResult, String> {
    // Safety ceiling on the positional scan. Set high enough that a full
    // pass is the norm; only pathologically frequent nodes hit it, and
    // when they do `truncated` surfaces it rather than silently capping.
    const MAX_NODE_OCCURRENCES: usize = 1_000_000;

    let lw = req.left_window.min(30);
    let rw = req.right_window.min(30);
    if lw == 0 && rw == 0 {
        return Err("collocation window must include at least one side".to_owned());
    }
    let layer: QueryLayer = req.layer.into();
    let limit = req.limit.clamp(1, 200);
    let span = (lw + rw) as u64;
    let t0 = Instant::now();

    let (collocates, node_freq, window_tokens, truncated) =
        with_corpus(state, &req.corpus_id, |index| {
            let scan = index
                .collocate_counts(&req.term, layer, lw, rw, MAX_NODE_OCCURRENCES)
                .map_err(|e| format!("collocate scan failed: {e:#}"))?;

            // Corpus size and collocate marginals are taken on the
            // surface (word) layer — collocates are surface forms read
            // from `body`, regardless of which layer the node matched on.
            let n = index
                .layer_total_tokens(corpust_index::QueryLayer::Word)
                .map_err(|e| format!("corpus size lookup failed: {e:#}"))?;

            // Bound the marginal lookups: score a generous top-by-raw-count
            // candidate pool rather than every distinct collocate type
            // (which can be tens of thousands for a frequent node).
            let pool = (limit * 8).max(200);
            // Exclude the node from its own collocate list (it co-occurs
            // with itself within the window, e.g. "the Jew … the Jew").
            let node_word = req.term.trim().to_lowercase();
            let mut cand: Vec<(String, u32, u32)> = scan
                .counts
                .iter()
                .filter(|(w, _)| **w != node_word)
                .map(|(w, &(l, r))| (w.clone(), l, r))
                .collect();
            cand.sort_by(|a, b| (b.1 + b.2).cmp(&(a.1 + a.2)).then_with(|| a.0.cmp(&b.0)));
            cand.truncate(pool);

            let words: Vec<String> = cand.iter().map(|(w, _, _)| w.clone()).collect();
            let f_coll = index
                .term_totals(&words, corpust_index::QueryLayer::Word)
                .map_err(|e| format!("collocate frequency lookup failed: {e:#}"))?;

            let mut collocates: Vec<CollocateDto> = cand
                .into_iter()
                .map(|(w, l, r)| {
                    let total = l + r;
                    let fc = f_coll.get(&w).copied().unwrap_or(0);
                    let s = corpust_index::assoc::scores(total as u64, scan.node_freq, fc, n, span);
                    CollocateDto {
                        word: w,
                        pos: String::new(),
                        left_count: l,
                        right_count: r,
                        total,
                        log_dice: s.log_dice,
                        mi: s.mi,
                        z: s.z,
                        dist: 0,
                    }
                })
                .collect();
            // Rank by log Dice (the UI default); the client re-sorts on
            // demand for the other measures.
            collocates.sort_by(|a, b| {
                b.log_dice
                    .partial_cmp(&a.log_dice)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.word.cmp(&b.word))
            });
            collocates.truncate(limit);

            Ok((
                collocates,
                scan.node_freq,
                scan.window_tokens,
                scan.truncated,
            ))
        })?;

    Ok(CollocatesResult {
        collocates,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        node_hits: node_freq.min(u32::MAX as u64) as u32,
        window_tokens: window_tokens.min(u32::MAX as u64) as u32,
        truncated,
    })
}

/// Async so the KWIC scan runs off the UI thread; a frequent term in a
/// large corpus would otherwise freeze the window while the loading
/// state is supposed to be showing.
#[tauri::command]
pub async fn run_kwic(app: AppHandle, req: KwicRequest) -> Result<KwicResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        run_kwic_inner(&state, &req)
    })
    .await
    .map_err(|e| format!("kwic task failed to join: {e}"))?
}

fn run_kwic_inner(state: &State<'_, AppState>, req: &KwicRequest) -> Result<KwicResult, String> {
    let context = if req.context == 0 {
        DEFAULT_CONTEXT
    } else {
        req.context
    };
    let limit = req.limit.max(1);
    let kreq = CoreKwicRequest::new(&req.term)
        .layer(req.layer.into())
        .context(context)
        .limit(limit);

    let t0 = Instant::now();
    let hits = with_corpus(state, &req.corpus_id, |index| {
        run_core_kwic(index, kreq).map_err(|e| format!("kwic failed: {e:#}"))
    })?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let truncated = hits.len() == limit;
    Ok(KwicResult {
        hits: hits
            .into_iter()
            .map(|h| KwicHitDto {
                doc_id: h.doc_id,
                path: h.path.to_string_lossy().into_owned(),
                hit_position: h.hit_position,
                left: h.left,
                hit: h.hit,
                right: h.right,
            })
            .collect(),
        elapsed_ms,
        truncated,
    })
}

/// List every document in a corpus with its path + token count. Backs
/// the CorpusDetail document table.
/// Async so the document-listing scan (reads every doc's stored fields)
/// stays off the UI thread on large corpora.
#[tauri::command]
pub async fn list_documents(
    app: AppHandle,
    corpus_id: String,
) -> Result<Vec<DocumentInfoDto>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        list_documents_inner(&state, &corpus_id)
    })
    .await
    .map_err(|e| format!("list documents task failed to join: {e}"))?
}

fn list_documents_inner(
    state: &State<'_, AppState>,
    corpus_id: &str,
) -> Result<Vec<DocumentInfoDto>, String> {
    with_corpus(state, corpus_id, |index| {
        let docs = index
            .list_documents()
            .map_err(|e| format!("listing documents: {e:#}"))?;
        Ok(docs
            .into_iter()
            .map(|d| DocumentInfoDto {
                doc_id: d.doc_id,
                path: d.path.to_string_lossy().into_owned(),
                token_count: d.token_count,
                title: d.title,
                author: d.author,
                year: d.year,
            })
            .collect())
    })
}

/// Corpus-wide top-N term frequencies on a layer. Backs the FrequencyView
/// word / POS tables.
/// Async so the term-dictionary scan (the fallback when no precomputed
/// sidecar exists) runs off the UI thread — Tauri puts sync commands on
/// the main thread, where switching the word/lemma/POS layer would
/// otherwise freeze the window. The frontend shows a loading state.
#[tauri::command]
pub async fn run_frequencies(
    app: AppHandle,
    req: FrequenciesRequest,
) -> Result<FrequenciesResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        run_frequencies_inner(&state, &req)
    })
    .await
    .map_err(|e| format!("frequencies task failed to join: {e}"))?
}

fn run_frequencies_inner(
    state: &State<'_, AppState>,
    req: &FrequenciesRequest,
) -> Result<FrequenciesResult, String> {
    let limit = req.limit.clamp(1, 1000);
    let layer: QueryLayer = req.layer.into();
    let t0 = Instant::now();
    // Prefer the precomputed sidecar written at build time; fall back to
    // a live term-dictionary scan for corpora built before it existed
    // (or if the request asks for more rows than were precomputed).
    let (rows, total_tokens) = match load_precomputed_frequencies(&req.corpus_id, layer, limit) {
        Some(table) => table,
        None => with_corpus(state, &req.corpus_id, |index| {
            index
                .frequencies(layer, limit)
                .map_err(|e| format!("frequencies failed: {e:#}"))
        })?,
    };
    let denom = total_tokens.max(1) as f64;
    let rows = rows
        .into_iter()
        .map(|(term, count)| FreqRow {
            term,
            count,
            pct: count as f64 / denom * 100.0,
        })
        .collect();
    Ok(FrequenciesResult {
        rows,
        total_tokens,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Per-document hit counts + a corpus-wide dispersion histogram for a
/// term. Backs the FrequencyView document table and dispersion strip.
/// Async for the same reason as [`run_frequencies`]: the per-term
/// postings walk that builds the dispersion histogram + per-doc counts
/// must stay off the UI thread.
#[tauri::command]
pub async fn run_term_distribution(
    app: AppHandle,
    req: TermDistRequest,
) -> Result<TermDistResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        run_term_distribution_inner(&state, &req)
    })
    .await
    .map_err(|e| format!("term distribution task failed to join: {e}"))?
}

fn run_term_distribution_inner(
    state: &State<'_, AppState>,
    req: &TermDistRequest,
) -> Result<TermDistResult, String> {
    let buckets = req.buckets.clamp(1, 1000);
    let t0 = Instant::now();
    let dist = with_corpus(state, &req.corpus_id, |index| {
        index
            .term_distribution(&req.term, req.layer.into(), buckets)
            .map_err(|e| format!("term distribution failed: {e:#}"))
    })?;
    Ok(TermDistResult {
        doc_counts: dist
            .doc_counts
            .into_iter()
            .map(|d| DocTermCountDto {
                doc_id: d.doc_id,
                path: d.path.to_string_lossy().into_owned(),
                hits: d.hits,
                token_count: d.token_count,
            })
            .collect(),
        dispersion: dist.dispersion,
        total_hits: dist.total_hits,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Per-collocate counts bucketed by signed distance from the node, for
/// the collocation-by-distance heatmap. Async — full positional scan.
#[tauri::command]
pub async fn run_collocate_distance(
    app: AppHandle,
    req: CollocateDistanceRequest,
) -> Result<CollocateDistanceResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        run_collocate_distance_inner(&state, &req)
    })
    .await
    .map_err(|e| format!("collocate distance task failed to join: {e}"))?
}

fn run_collocate_distance_inner(
    state: &State<'_, AppState>,
    req: &CollocateDistanceRequest,
) -> Result<CollocateDistanceResult, String> {
    const MAX_NODE_OCCURRENCES: usize = 1_000_000;
    let lw = req.left_window.min(15);
    let rw = req.right_window.min(15);
    if lw == 0 && rw == 0 {
        return Err("distance window must include at least one side".to_owned());
    }
    let layer: QueryLayer = req.layer.into();
    let limit = req.limit.clamp(1, 60);
    let t0 = Instant::now();

    let (offsets, mut rows, node_freq, truncated) = with_corpus(state, &req.corpus_id, |index| {
        let prof = index
            .collocate_by_distance(&req.term, layer, lw, rw, MAX_NODE_OCCURRENCES)
            .map_err(|e| format!("distance scan failed: {e:#}"))?;
        let node_word = req.term.trim().to_lowercase();
        let rows: Vec<DistanceRow> = prof
            .rows
            .into_iter()
            .filter(|(word, _)| *word != node_word) // exclude the node itself
            .map(|(word, counts)| {
                let total = counts.iter().sum();
                DistanceRow {
                    word,
                    total,
                    counts,
                }
            })
            .collect();
        Ok((prof.offsets, rows, prof.node_freq, prof.truncated))
    })?;

    // Keep the busiest collocates; the heatmap can't show thousands.
    rows.sort_by(|a, b| b.total.cmp(&a.total).then_with(|| a.word.cmp(&b.word)));
    rows.truncate(limit);

    Ok(CollocateDistanceResult {
        offsets,
        rows,
        node_hits: node_freq.min(u32::MAX as u64) as u32,
        truncated,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Re-expand the context around a single KWIC hit to a wider window.
/// Backs the ContextDrawer.
/// Async because locating the document is a linear scan over stored doc
/// ids; on a large corpus that's enough to stutter the UI if it ran on
/// the main thread when the drawer opens.
#[tauri::command]
pub async fn expand_context(app: AppHandle, req: ExpandRequest) -> Result<ExpandedContext, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        expand_context_inner(&state, &req)
    })
    .await
    .map_err(|e| format!("expand context task failed to join: {e}"))?
}

fn expand_context_inner(
    state: &State<'_, AppState>,
    req: &ExpandRequest,
) -> Result<ExpandedContext, String> {
    let context = req.context.clamp(1, 500);
    let found = with_corpus(state, &req.corpus_id, |index| {
        index
            .context_at(req.doc_id, req.position, context)
            .map_err(|e| format!("context lookup failed: {e:#}"))
    })?;
    // Resolve the path separately so the drawer can show a title even if
    // the position fell out of range.
    let path = with_corpus(state, &req.corpus_id, |index| {
        Ok(index
            .list_documents()
            .map_err(|e| format!("listing documents: {e:#}"))?
            .into_iter()
            .find(|d| d.doc_id == req.doc_id)
            .map(|d| d.path.to_string_lossy().into_owned())
            .unwrap_or_default())
    })?;
    match found {
        Some((before, hit, after, token_count)) => Ok(ExpandedContext {
            doc_id: req.doc_id,
            path,
            before,
            hit,
            after,
            token_count,
        }),
        None => Err(format!(
            "no token at position {} in doc {}",
            req.position, req.doc_id
        )),
    }
}

/// Runs the build on a worker thread so the UI event loop stays
/// responsive. Tauri routes sync `fn` commands onto the main thread
/// by default — long-running work in that context freezes the
/// window. Making the command `async fn` + punting to
/// `spawn_blocking` keeps both sides happy: no UI freeze, and the
/// file/I-O/annotator code inside stays synchronous.
#[tauri::command]
pub async fn build_index(app: AppHandle, req: BuildRequest) -> Result<CorpusMeta, String> {
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
    let corpus_dir =
        paths::corpus_dir(&slug).map_err(|e| format!("resolving corpus dir: {e:#}"))?;
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

    let indexing_phase = if req.annotate {
        "annotating"
    } else {
        "indexing"
    };
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
                emit_progress(
                    app,
                    started,
                    indexing_phase,
                    seen as u64,
                    Some(doc_count as u64),
                );
                last_emitted = seen;
                last_instant = Instant::now();
            }
        })
        .map_err(|e| format!("indexing failed: {e:#}"))?;
    let build_ms = t_build.elapsed().as_millis() as u64;
    emit_progress(
        app,
        started,
        "committing",
        doc_count as u64,
        Some(doc_count as u64),
    );

    let mut meta = CorpusMeta::stub(slug.clone(), name, out_path.to_string_lossy().into_owned());
    meta.source_path = source_path.to_string_lossy().into_owned();
    meta.annotated = req.annotate;
    meta.doc_count = doc_count as u64;
    // Rough byte-based token approximation — a proper count needs
    // an aggregation pass over the index. Good enough for the UI's
    // "built: N tokens" header for now.
    meta.token_count = (byte_count / 6) as u64;
    meta.avg_doc_len = if doc_count > 0 {
        (byte_count / doc_count) as u64
    } else {
        0
    };
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

    // Precompute per-layer frequency tables so FrequencyView serves from
    // a sidecar instead of re-scanning the term dictionary on every open.
    // Best-effort — the query path falls back to a live scan if absent.
    write_frequency_sidecar(&index, &corpus_dir);

    emit_progress(
        app,
        started,
        "done",
        doc_count as u64,
        Some(doc_count as u64),
    );
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

/// Precompute and persist the per-layer frequency tables next to the
/// index. Best-effort: any failure only forfeits the speedup (queries
/// fall back to a live scan), so it never aborts a build — it logs.
fn write_frequency_sidecar(index: &CorpusIndex, corpus_dir: &Path) {
    use corpust_io::freq::{FreqTables, LayerFreq, PRECOMPUTE_LIMIT, write_freq_file};
    let af = match index.all_frequencies(PRECOMPUTE_LIMIT) {
        Ok(af) => af,
        Err(e) => {
            eprintln!("warning: couldn't precompute frequencies: {e:#}");
            return;
        }
    };
    let tables = FreqTables {
        limit: PRECOMPUTE_LIMIT,
        word: LayerFreq::from_table(af.word),
        lemma: LayerFreq::from_table(af.lemma),
        pos: LayerFreq::from_table(af.pos),
    };
    let path = corpus_dir.join("frequencies.json");
    if let Err(e) = write_freq_file(&path, &tables) {
        eprintln!("warning: couldn't write {}: {e:#}", path.display());
    }
}

/// Serve a frequency request from the precomputed sidecar, or `None` to
/// signal the caller should fall back to a live scan. Returns `None`
/// when the sidecar is absent/unreadable or was computed with fewer rows
/// than `limit` asks for.
fn load_precomputed_frequencies(
    slug: &str,
    layer: QueryLayer,
    limit: usize,
) -> Option<(Vec<(String, u64)>, u64)> {
    let path = paths::freq_path(slug).ok()?;
    load_precomputed_from(&path, layer, limit)
}

/// Path-based core of [`load_precomputed_frequencies`], split out so it
/// can be tested without touching the global `CORPUST_DATA_ROOT`.
fn load_precomputed_from(
    path: &Path,
    layer: QueryLayer,
    limit: usize,
) -> Option<(Vec<(String, u64)>, u64)> {
    if !path.exists() {
        return None;
    }
    let tables = corpust_io::freq::read_freq_file(path).ok()?;
    if tables.limit < limit {
        return None;
    }
    let layer_freq = match layer {
        QueryLayer::Word => tables.word,
        QueryLayer::Lemma => tables.lemma,
        QueryLayer::Pos => tables.pos,
    };
    let mut rows = layer_freq.rows;
    rows.truncate(limit);
    Some((rows, layer_freq.total))
}

/// Open an existing corpus from disk by slug. Returns the tantivy
/// handle plus the persisted metadata.
fn load_from_disk(slug: &str) -> Result<(CorpusIndex, CorpusMeta), String> {
    let corpus_dir =
        paths::corpus_dir(slug).map_err(|e| format!("resolving corpus dir for {slug}: {e:#}"))?;
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

// Metadata sidecar helpers live in `corpust_io::metadata` and are
// re-exported above for the Tauri command bodies that need them.
use corpust_io::metadata::{dir_size, iso_now, write_metadata_file};

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
    let abbr = bundle.join("lib").join(format!("{language}-abbreviations"));
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
    for rel in [
        "resources/treetagger",
        "../resources/treetagger",
        "../../resources/treetagger",
    ] {
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
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tmp_file(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "corpust-meta-{}-{}.json",
            name,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
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
        assert_eq!(
            read_back.annotator.as_deref(),
            Some("treetagger-rs-english")
        );
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
            err.to_string()
                .contains("unsupported metadata schema version"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn precomputed_frequencies_serve_truncate_and_fallback() {
        use corpust_io::freq::{FreqTables, LayerFreq, PRECOMPUTE_LIMIT, write_freq_file};

        let path = tmp_file("freq-serve");
        let tables = FreqTables {
            limit: PRECOMPUTE_LIMIT,
            word: LayerFreq {
                rows: vec![("the".into(), 10), ("cat".into(), 5), ("dog".into(), 3)],
                total: 18,
            },
            lemma: LayerFreq::default(),
            pos: LayerFreq {
                rows: vec![("NN".into(), 8)],
                total: 18,
            },
        };
        write_freq_file(&path, &tables).unwrap();

        // Serves the word layer, truncated to the requested limit.
        let (rows, total) = load_precomputed_from(&path, QueryLayer::Word, 2).unwrap();
        assert_eq!(rows, vec![("the".into(), 10), ("cat".into(), 5)]);
        assert_eq!(total, 18);

        // POS layer comes through; total is the field grand total.
        let (pos_rows, _) = load_precomputed_from(&path, QueryLayer::Pos, 10).unwrap();
        assert_eq!(pos_rows, vec![("NN".into(), 8)]);

        // Lemma layer is empty for an unannotated corpus — served, not a fallback.
        let (lemma_rows, _) = load_precomputed_from(&path, QueryLayer::Lemma, 10).unwrap();
        assert!(lemma_rows.is_empty());

        std::fs::remove_file(&path).ok();

        // Missing sidecar → None, signalling a live-scan fallback.
        assert!(load_precomputed_from(&path, QueryLayer::Word, 2).is_none());
    }

    #[test]
    fn precomputed_frequencies_falls_back_when_limit_exceeds_precompute() {
        use corpust_io::freq::{FreqTables, LayerFreq, write_freq_file};

        let path = tmp_file("freq-toosmall");
        let tables = FreqTables {
            limit: 2, // precomputed only top-2
            word: LayerFreq {
                rows: vec![("the".into(), 10), ("cat".into(), 5)],
                total: 18,
            },
            ..Default::default()
        };
        write_freq_file(&path, &tables).unwrap();

        // Asking for more than was precomputed forces a fallback.
        assert!(load_precomputed_from(&path, QueryLayer::Word, 5).is_none());
        // Within the precomputed depth it still serves.
        assert!(load_precomputed_from(&path, QueryLayer::Word, 2).is_some());

        std::fs::remove_file(&path).ok();
    }
}
