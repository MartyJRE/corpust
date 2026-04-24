// Top-level composition. Holds all view / corpus / query / overlay state.
// TODO: replace in-memory `corpora` and `pickHits` with Tauri IPC —
//   listCorpora()  → corpora
//   runKwic(req)   → result
//   buildIndex(…)  → stream progress into BuildDialog

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Sidebar } from "@/components/chrome/Sidebar";
import { StatusBar } from "@/components/chrome/StatusBar";
import { TitleStrip } from "@/components/chrome/TitleStrip";
import { ViewTabs } from "@/components/chrome/ViewTabs";
import { QueryBar, type Filter } from "@/components/query/QueryBar";
import { KwicTable } from "@/components/kwic/KwicTable";
import { HitDensityGutter } from "@/components/kwic/HitDensityGutter";
import { ContextDrawer } from "@/components/kwic/ContextDrawer";
import { CollocationsView } from "@/components/analyses/CollocationsView";
import { FrequencyView } from "@/components/analyses/FrequencyView";
import { CorpusDetail } from "@/components/analyses/CorpusDetail";
import { SettingsView } from "@/components/analyses/SettingsView";
import { Onboarding } from "@/components/analyses/Onboarding";
import { CommandPalette, type CommandDef } from "@/components/overlays/CommandPalette";
import { BuildDialog } from "@/components/overlays/BuildDialog";
import { CORPORA, RECENT_QUERIES, pickHits } from "@/data";
import { makeDensity } from "@/lib/utils";
import { inTauri, listCorpora, runCollocates, runKwic as runKwicTauri } from "@/lib/tauri";
import type { Collocate } from "@/types";
import type {
  CorpusMeta,
  KwicHit,
  KwicResult,
  MainView,
  QueryLayer,
  RecentQuery,
  SortMode,
  SubView,
} from "@/types";

export function App() {
  const [corpora, setCorpora] = useState<CorpusMeta[]>(CORPORA);
  const [activeId, setActiveId] = useState<string | null>("gut-en");
  const [view, setView] = useState<MainView>("search");
  const [subview, setSubview] = useState<SubView>("kwic");
  const [layer, setLayer] = useState<QueryLayer>("word");
  const [term, setTerm] = useState("linguistic");
  const [result, setResult] = useState<KwicResult | null>(null);
  const [collocates, setCollocates] = useState<Collocate[] | null>(null);
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
  const [filters, setFilters] = useState<Filter[]>([{ key: "year", label: "year: 1800–1950" }]);
  const scrollPct = 0.18;

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
  const kwicReqRef = useRef(0);
  const runKwic = () => {
    if (!activeCorpus || !term.trim()) return;
    const myId = ++kwicReqRef.current;
    setLoading(true);
    const isRealCorpus = inTauri() && !CORPORA.some((c) => c.id === activeCorpus.id);
    if (isRealCorpus) {
      runKwicTauri({
        corpusId: activeCorpus.id,
        term: term.trim(),
        layer,
        context: 8,
        limit: 200,
      })
        .then((r) => {
          if (myId !== kwicReqRef.current) return;
          // The Tauri backend returns hits with a numeric doc_id +
          // file path; the frontend's KwicHit shape carries docId
          // (string) and no path. Adapt the shape inline until the
          // types converge.
          const hits: KwicHit[] = r.hits.map((h, i) => ({
            docId: String(h.docId),
            pos: i,
            left: h.left,
            hit: h.hit,
            right: h.right,
          }));
          setResult({ hits, elapsedMs: r.elapsedMs, truncated: r.truncated });
          setSelected(null);
        })
        .catch((e) => {
          console.error("runKwic failed:", e);
          if (myId === kwicReqRef.current) {
            setResult({ hits: [], elapsedMs: 0, truncated: false });
            setSelected(null);
          }
        })
        .finally(() => {
          if (myId === kwicReqRef.current) setLoading(false);
        });
      return;
    }
    // Fallback: fixture data for the non-Tauri / pre-built-corpus case.
    window.setTimeout(() => {
      if (myId !== kwicReqRef.current) return;
      const hits = pickHits(activeCorpus.id, term.trim(), layer);
      const elapsedMs = 0.2 + Math.random() * 1.6;
      setResult({ hits, elapsedMs, truncated: hits.length >= 1000 });
      setSelected(null);
      setLoading(false);
    }, 40);
  };
  // Explicit Enter-key / "Run" button handler. Same path as the
  // effects — the request-id guard handles any overlap.
  const run = runKwic;

  useEffect(() => {
    runKwic();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layer, activeCorpus]);

  useEffect(() => {
    const t = window.setTimeout(runKwic, 100);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [term]);

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
    if (subview !== "coll" || !activeCorpus || !term.trim()) return;
    if (!inTauri() || CORPORA.some((c) => c.id === activeCorpus.id)) {
      setCollocates(null);
      return;
    }
    const myId = ++collReqRef.current;
    runCollocates({
      corpusId: activeCorpus.id,
      term: term.trim(),
      layer,
      leftWindow: collLeft,
      rightWindow: collRight,
      limit: 60,
    })
      .then((r) => {
        if (myId === collReqRef.current) setCollocates(r.collocates);
      })
      .catch((e) => {
        console.error("runCollocates failed:", e);
        if (myId === collReqRef.current) setCollocates([]);
      });
  };

  useEffect(() => {
    fetchCollocates();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [subview, activeCorpus, layer, collLeft, collRight]);

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
  }, [activeCorpus, result, selected, paletteOpen, buildOpen]);

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
          filters={filters}
          onRemoveFilter={(k) => setFilters((fs) => fs.filter((f) => f.key !== k))}
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
              />
              <HitDensityGutter density={density} scrollPct={scrollPct} onJump={() => {}} />
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
              data={collocates}
              leftWindow={collLeft}
              rightWindow={collRight}
              onWindowChange={(l, r) => {
                setCollLeft(l);
                setCollRight(r);
              }}
            />
          )}
          {subview === "freq" && activeCorpus && (
            <FrequencyView corpus={activeCorpus} term={term} />
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
            memory={0.42}
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
