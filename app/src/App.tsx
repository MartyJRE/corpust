// Top-level composition. Holds all view / corpus / query / overlay state.
// Real corpora are served over Tauri IPC (listCorpora / runKwic /
// buildIndex …); the baked-in `@/data` fixtures are the fallback shown
// when no real corpus is loaded (vite preview or the demo set).

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Sidebar } from "@/components/chrome/Sidebar";
import { StatusBar } from "@/components/chrome/StatusBar";
import { TitleStrip } from "@/components/chrome/TitleStrip";
import { ViewTabs } from "@/components/chrome/ViewTabs";
import { QueryBar } from "@/components/query/QueryBar";
import { KwicTable } from "@/components/kwic/KwicTable";
import { HitDensityGutter } from "@/components/kwic/HitDensityGutter";
import { ContextDrawer } from "@/components/kwic/ContextDrawer";
import { CollocationsView } from "@/components/analyses/CollocationsView";
import { CollocationDistance } from "@/components/analyses/CollocationDistance";
import { FrequencyOverTime } from "@/components/analyses/FrequencyOverTime";
import { WordTree } from "@/components/analyses/WordTree";
import { FrequencyView } from "@/components/analyses/FrequencyView";
import { CorpusDetail } from "@/components/analyses/CorpusDetail";
import { SettingsView } from "@/components/analyses/SettingsView";
import { Onboarding } from "@/components/analyses/Onboarding";
import { CommandPalette, type CommandDef } from "@/components/overlays/CommandPalette";
import { BuildDialog } from "@/components/overlays/BuildDialog";
import { CORPORA, RECENT_QUERIES, pickHits } from "@/data";
import { concordanceCsv, concordanceJson, saveText, slug } from "@/lib/export";
import { normalizeFilter } from "@/lib/filter";
import { makeDensity } from "@/lib/utils";
import { hasLiveData, inTauri, isFixtureCorpus, listCorpora, runCollocates, runKwic as runKwicTauri } from "@/lib/tauri";
import type { Collocate } from "@/types";
import type {
  CorpusMeta,
  DocFilter,
  KwicHit,
  KwicResult,
  MainView,
  QueryLayer,
  RecentQuery,
  SortMode,
  SubView,
} from "@/types";

/** Concordance lines fetched per page. */
const KWIC_PAGE = 200;

export function App() {
  const [corpora, setCorpora] = useState<CorpusMeta[]>(CORPORA);
  const [activeId, setActiveId] = useState<string | null>("gut-en");
  const [view, setView] = useState<MainView>("search");
  const [subview, setSubview] = useState<SubView>("kwic");
  const [layer, setLayer] = useState<QueryLayer>("word");
  const [term, setTerm] = useState("linguistic");
  const [result, setResult] = useState<KwicResult | null>(null);
  const [collocates, setCollocates] = useState<Collocate[] | null>(null);
  const [collLoading, setCollLoading] = useState(false);
  const [collTruncated, setCollTruncated] = useState(false);
  const [collLeft, setCollLeft] = useState(5);
  const [collRight, setCollRight] = useState(5);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<KwicHit | null>(null);
  const [sortMode, setSortMode] = useState<SortMode>("right1");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const onSortChange = useCallback(
    (mode: SortMode) => {
      if (mode === sortMode) {
        setSortDir((d) => (d === "asc" ? "desc" : "asc"));
      } else {
        setSortMode(mode);
        setSortDir("asc");
      }
    },
    [sortMode],
  );
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [buildOpen, setBuildOpen] = useState(false);
  // Document-metadata filter applied to every query. `queryFilter` is the
  // normalized form (empty dimensions dropped, or undefined when nothing
  // is set) that travels to the backend.
  const [filter, setFilter] = useState<DocFilter>({});
  const queryFilter = useMemo(() => normalizeFilter(filter), [filter]);
  // Concordance scroll position (0–1), tracked from the KWIC column so the
  // hit-density gutter's thumb reflects where you actually are, and clicks
  // on the gutter scroll there.
  const kwicScrollRef = useRef<HTMLDivElement>(null);
  const [scrollPct, setScrollPct] = useState(0);

  // Refresh the corpora list from disk via the Tauri backend. Real
  // (persisted) corpora land at the top; baked-in fixtures stay below
  // as the always-available demo set.
  const refreshCorpora = useCallback(async () => {
    if (!inTauri()) return;
    try {
      const saved = await listCorpora();
      setCorpora((prev) => {
        const fixtures = prev.filter((c) => CORPORA.some((f) => f.id === c.id));
        const dedupedSaved = saved.filter(
          (c) => !fixtures.some((f) => f.id === c.id),
        );
        return [...dedupedSaved, ...fixtures];
      });
      // If the current selection is a fixture and we just loaded a
      // real corpus, switch to the first real one so the user lands
      // on their own data instead of the demo.
      if (saved.length > 0) {
        setActiveId((current) => {
          if (current && saved.some((c) => c.id === current)) return current;
          return saved[0].id;
        });
      }
    } catch (e) {
      console.error("listCorpora failed:", e);
    }
  }, []);

  useEffect(() => {
    void refreshCorpora();
  }, [refreshCorpora]);

  const activeCorpus = useMemo(
    () => corpora.find((c) => c.id === activeId) ?? null,
    [corpora, activeId],
  );
  const density = useMemo(() => makeDensity(result ? result.hits : []), [result]);

  // KWIC live-updates the same way collocates do. Typing debounces by
  // 100 ms; layer / corpus switches refetch immediately; a request-id
  // counter drops stale responses so the freshest one wins. We don't
  // wipe `result` on refetch — the stale table stays visible so the
  // UI doesn't strobe on every keystroke.
  // One fetch path for the concordance, keyed on (corpus, term, layer).
  // A request-id guard drops superseded responses, so a real corpus can
  // never end up showing a previous (fixture) corpus's stale hits.
  const kwicReqRef = useRef(0);
  const fetchKwic = useCallback(
    (corpus: CorpusMeta | null, q: string, lyr: QueryLayer, offset: number) => {
      if (!corpus || !q.trim()) {
        setResult(null);
        setLoading(false);
        return;
      }
      const myId = ++kwicReqRef.current;
      setLoading(true);
      // Fixture corpora (or non-Tauri preview) use the baked-in demo hits.
      if (!inTauri() || isFixtureCorpus(corpus.id)) {
        const hits = pickHits(corpus.id, q.trim(), lyr);
        if (myId === kwicReqRef.current) {
          setResult({ hits, elapsedMs: 0.2 + Math.random() * 1.6, truncated: false, total: hits.length, offset: 0 });
          setSelected(null);
          setLoading(false);
        }
        return;
      }
      runKwicTauri({ corpusId: corpus.id, term: q.trim(), layer: lyr, context: 8, limit: KWIC_PAGE, offset, filter: queryFilter })
        .then((r) => {
          if (myId !== kwicReqRef.current) return;
          const hits: KwicHit[] = r.hits.map((h, i) => ({
            docId: String(h.docId),
            pos: i,
            hitPos: h.hitPosition,
            left: h.left,
            hit: h.hit,
            right: h.right,
          }));
          setResult({ hits, elapsedMs: r.elapsedMs, truncated: r.truncated, total: r.total, offset: r.offset });
          setSelected(null);
        })
        .catch((e) => {
          if (myId !== kwicReqRef.current) return;
          console.error("runKwic failed:", e);
          setResult({ hits: [], elapsedMs: 0, truncated: false, total: 0, offset: 0 });
          setSelected(null);
        })
        .finally(() => {
          if (myId === kwicReqRef.current) setLoading(false);
        });
    },
    [queryFilter],
  );

  // Auto-fetch on corpus / term / layer change — always resets to the
  // first page (debounced so typing coalesces). The guard drops stale
  // responses.
  useEffect(() => {
    const t = window.setTimeout(() => fetchKwic(activeCorpus, term, layer, 0), 100);
    return () => window.clearTimeout(t);
  }, [activeCorpus, term, layer, fetchKwic]);

  // Jump to a concordance page (offset in hits) — fetches immediately.
  const goToPage = (offset: number) => fetchKwic(activeCorpus, term, layer, offset);

  // Explicit Enter / "Run" — fetch the first page immediately.
  const run = () => fetchKwic(activeCorpus, term, layer, 0);

  // Keep `scrollPct` in sync with the concordance column so the density
  // gutter's thumb tracks the real scroll position. Re-attaches whenever
  // the table remounts (result/subview change swap the scroll element).
  useEffect(() => {
    const el = kwicScrollRef.current;
    if (!el) return;
    const onScroll = () => {
      const max = el.scrollHeight - el.clientHeight;
      setScrollPct(max > 0 ? el.scrollTop / max : 0);
    };
    onScroll();
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [result, subview, view]);

  // Click on the density gutter → scroll the concordance to that fraction.
  const onDensityJump = (pct: number) => {
    const el = kwicScrollRef.current;
    if (!el) return;
    el.scrollTo({ top: pct * (el.scrollHeight - el.clientHeight), behavior: "smooth" });
  };

  // Export the current concordance page. CSV is flat (n, doc, position,
  // left, node, right); JSON carries the query context + any lemma/POS.
  const exportConcordance = useCallback(
    (format: "csv" | "json") => {
      if (!result || result.hits.length === 0) return;
      const base = `${slug(activeCorpus?.name ?? "concordance")}-${slug(term || "query")}`;
      if (format === "csv") {
        void saveText(`${base}.csv`, concordanceCsv(result), "text/csv");
      } else {
        void saveText(`${base}.json`, concordanceJson(result, activeCorpus, term, layer), "application/json");
      }
    },
    [result, activeCorpus, term, layer],
  );

  // Fetch real collocates when the Collocations view is active on a
  // real (backend-registered) corpus. Fixture corpora keep whatever
  // data.ts ships.
  //
  // Split into two effects: button-click triggers (L/R window, layer,
  // subview switch) refetch immediately so the UI feels snappy. Query
  // text changes go through a 100 ms debounce so bursty typing
  // coalesces into a single backend call. A request-id counter drops
  // out-of-order responses so a stale answer can't clobber a fresh one.
  const collReqRef = useRef(0);
  const fetchCollocates = () => {
    if (subview !== "coll" || !activeCorpus) return;
    // Empty query: clear stale collocates instead of leaving the last
    // result on screen with a blank node.
    if (!term.trim()) {
      collReqRef.current++;
      setCollocates(inTauri() && !CORPORA.some((c) => c.id === activeCorpus.id) ? [] : null);
      setCollLoading(false);
      setCollTruncated(false);
      return;
    }
    if (!inTauri() || CORPORA.some((c) => c.id === activeCorpus.id)) {
      setCollocates(null);
      setCollLoading(false);
      setCollTruncated(false);
      return;
    }
    const myId = ++collReqRef.current;
    setCollLoading(true);
    runCollocates({
      corpusId: activeCorpus.id,
      term: term.trim(),
      layer,
      leftWindow: collLeft,
      rightWindow: collRight,
      limit: 60,
      filter: queryFilter,
    })
      .then((r) => {
        if (myId !== collReqRef.current) return;
        setCollocates(r.collocates);
        setCollTruncated(r.truncated);
        setCollLoading(false);
      })
      .catch((e) => {
        console.error("runCollocates failed:", e);
        if (myId !== collReqRef.current) return;
        setCollocates([]);
        setCollTruncated(false);
        setCollLoading(false);
      });
  };

  useEffect(() => {
    fetchCollocates();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [subview, activeCorpus, layer, collLeft, collRight, queryFilter]);

  useEffect(() => {
    const t = window.setTimeout(fetchCollocates, 100);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [term]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const meta = e.metaKey || e.ctrlKey;
      if (meta && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen(true);
      } else if (meta && e.key.toLowerCase() === "b") {
        e.preventDefault();
        setBuildOpen(true);
      } else if (meta && e.key === "1") {
        e.preventDefault();
        setLayer("word");
      } else if (meta && e.key === "2") {
        e.preventDefault();
        if (activeCorpus?.annotated) setLayer("lemma");
      } else if (meta && e.key === "3") {
        e.preventDefault();
        if (activeCorpus?.annotated) setLayer("pos");
      } else if (meta && e.key.toLowerCase() === "e") {
        e.preventDefault();
        exportConcordance("csv");
      } else if (e.key === "Escape") {
        setSelected(null);
      } else if ((e.key === "j" || e.key === "k") && result && !paletteOpen && !buildOpen) {
        const hits = result.hits;
        if (!hits.length) return;
        const idx = selected
          ? hits.findIndex((h) => h.docId === selected.docId && h.pos === selected.pos)
          : -1;
        const next =
          e.key === "j"
            ? Math.min(hits.length - 1, idx + 1)
            : Math.max(0, idx === -1 ? 0 : idx - 1);
        setSelected(hits[next]);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [activeCorpus, result, selected, paletteOpen, buildOpen, exportConcordance]);

  const runRecent = (q: RecentQuery) => {
    setLayer(q.layer);
    setTerm(q.term);
    setActiveId(q.corpus);
    setView("search");
    setSubview("kwic");
  };

  const onRunCmd = (cmd: CommandDef) => {
    if (cmd.id === "build") setBuildOpen(true);
    else if (cmd.id === "open") void refreshCorpora();
    else if (cmd.id === "detail") setView("corpus");
    else if (cmd.id === "layer-word") {
      setLayer("word");
    } else if (cmd.id === "layer-lemma") {
      if (activeCorpus?.annotated) {
        setLayer("lemma");
        setTerm("go");
      }
    } else if (cmd.id === "layer-pos") {
      if (activeCorpus?.annotated) {
        setLayer("pos");
        setTerm("NN");
      }
    } else if (cmd.id === "clear") {
      setTerm("");
      setResult(null);
    } else if (cmd.id === "view-kwic") {
      setView("search");
      setSubview("kwic");
    } else if (cmd.id === "view-coll") {
      setView("search");
      setSubview("coll");
    } else if (cmd.id === "view-freq") {
      setView("search");
      setSubview("freq");
    } else if (cmd.id === "export-csv") {
      exportConcordance("csv");
    } else if (cmd.id === "export-json") {
      exportConcordance("json");
    }
  };

  // Drawer nav
  const hitList = result ? result.hits : [];
  const selIdx = selected
    ? hitList.findIndex((h) => h.docId === selected.docId && h.pos === selected.pos)
    : -1;
  const onPrev = () => {
    if (selIdx > 0) setSelected(hitList[selIdx - 1]);
  };
  const onNext = () => {
    if (selIdx >= 0 && selIdx < hitList.length - 1) setSelected(hitList[selIdx + 1]);
  };

  // Empty state — no corpora
  if (corpora.length === 0) {
    return (
      <div className="cx-app">
        <TitleStrip view={view} onView={setView} />
        <div className="cx-body">
          <Onboarding
            onBuild={() => setBuildOpen(true)}
            onOpen={() => void refreshCorpora()}
            onSample={() => setCorpora(CORPORA)}
          />
        </div>
        <BuildDialog
          open={buildOpen}
          onClose={() => setBuildOpen(false)}
          onBuilt={(c) => {
            setCorpora([c]);
            setActiveId(c.id);
          }}
        />
      </div>
    );
  }

  const mainContent = () => {
    if (view === "settings") return <SettingsView />;
    if (view === "corpus" && activeCorpus) {
      return <CorpusDetail corpus={activeCorpus} onDismiss={() => setView("search")} />;
    }

    return (
      <>
        <QueryBar
          layer={layer}
          term={term}
          onLayer={setLayer}
          onTerm={setTerm}
          onRun={run}
          disabled={!activeCorpus}
          annotated={!!activeCorpus?.annotated}
          onOpenPalette={() => setPaletteOpen(true)}
          filter={filter}
          onFilterChange={setFilter}
          filterable={!!activeCorpus && hasLiveData(activeCorpus.id)}
        />
        <ViewTabs view={subview} onView={setSubview} result={result} />
        <div className="cx-results-wrap">
          {subview === "kwic" && (
            <>
              <KwicTable
                result={result}
                loading={loading}
                layer={layer}
                sortMode={sortMode}
                sortDir={sortDir}
                onSort={onSortChange}
                selected={selected}
                onSelect={setSelected}
                pageSize={KWIC_PAGE}
                onPage={goToPage}
                onExport={exportConcordance}
                scrollRef={kwicScrollRef}
              />
              <HitDensityGutter density={density} scrollPct={scrollPct} onJump={onDensityJump} />
              {selected && activeCorpus && (
                <ContextDrawer
                  hit={selected}
                  corpus={activeCorpus}
                  onClose={() => setSelected(null)}
                  onPrev={onPrev}
                  onNext={onNext}
                />
              )}
            </>
          )}
          {subview === "coll" && activeCorpus && (
            <CollocationsView
              corpus={activeCorpus}
              term={term}
              layer={layer}
              data={collocates}
              loading={collLoading}
              truncated={collTruncated}
              leftWindow={collLeft}
              rightWindow={collRight}
              onWindowChange={(l, r) => {
                setCollLeft(l);
                setCollRight(r);
              }}
            />
          )}
          {subview === "freq" && activeCorpus && (
            <FrequencyView corpus={activeCorpus} term={term} filter={queryFilter} />
          )}
          {subview === "tree" && activeCorpus && (
            <WordTree corpusName={activeCorpus.name} term={term} result={result} loading={loading} />
          )}
          {subview === "dist" && activeCorpus && (
            <CollocationDistance corpus={activeCorpus} term={term} layer={layer} filter={queryFilter} />
          )}
          {subview === "time" && activeCorpus && (
            <FrequencyOverTime corpus={activeCorpus} term={term} layer={layer} filter={queryFilter} />
          )}
        </div>
      </>
    );
  };

  return (
    <div className="cx-app">
      <TitleStrip view={view} onView={setView} />
      <div className="cx-body">
        <Sidebar
          corpora={corpora}
          activeId={activeId}
          onSelect={(id) => {
            setActiveId(id);
            setView("search");
          }}
          onOpen={() => void refreshCorpora()}
          onBuild={() => setBuildOpen(true)}
          recent={RECENT_QUERIES}
          onRunRecent={runRecent}
        />
        <main className="cx-main">
          {mainContent()}
          <StatusBar
            corpus={activeCorpus}
            result={view === "search" ? result : null}
            layer={layer}
          />
        </main>
      </div>

      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} onRun={onRunCmd} />
      <BuildDialog
        open={buildOpen}
        onClose={() => setBuildOpen(false)}
        onBuilt={(c) => {
          setCorpora((xs) => [c, ...xs]);
          setActiveId(c.id);
        }}
      />
    </div>
  );
}
