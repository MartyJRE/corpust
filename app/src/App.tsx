// Top-level composition. Holds all view / corpus / query / overlay state.
// TODO: replace in-memory `corpora` and `pickHits` with Tauri IPC —
//   listCorpora()  → corpora
//   runKwic(req)   → result
//   buildIndex(…)  → stream progress into BuildDialog

import { useCallback, useEffect, useMemo, useState } from "react";
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
import { inTauri, runCollocates, runKwic as runKwicTauri } from "@/lib/tauri";
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
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<KwicHit | null>(null);
  const [sortMode, setSortMode] = useState<SortMode>("right1");
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [buildOpen, setBuildOpen] = useState(false);
  const [filters, setFilters] = useState<Filter[]>([{ key: "year", label: "year: 1800–1950" }]);
  const scrollPct = 0.18;

  const activeCorpus = useMemo(
    () => corpora.find((c) => c.id === activeId) ?? null,
    [corpora, activeId],
  );
  const density = useMemo(() => makeDensity(result ? result.hits : []), [result]);

  const run = useCallback(() => {
    if (!activeCorpus || !term.trim()) return;
    setLoading(true);
    setResult(null);
    const isRealCorpus = inTauri() && activeCorpus.id.startsWith("corpus-");
    if (isRealCorpus) {
      runKwicTauri({
        corpusId: activeCorpus.id,
        term: term.trim(),
        layer,
        context: 8,
        limit: 200,
      })
        .then((r) => {
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
        })
        .catch((e) => {
          console.error("runKwic failed:", e);
          setResult({ hits: [], elapsedMs: 0, truncated: false });
        })
        .finally(() => setLoading(false));
      return;
    }
    // Fallback: fixture data for the non-Tauri / pre-built-corpus case.
    window.setTimeout(() => {
      const hits = pickHits(activeCorpus.id, term.trim(), layer);
      const elapsedMs = 0.2 + Math.random() * 1.6;
      setResult({ hits, elapsedMs, truncated: hits.length >= 1000 });
      setLoading(false);
    }, 40);
  }, [activeCorpus, term, layer]);

  // Initial query on mount
  useEffect(() => {
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch real collocates when the Collocations view is active on a
  // real (backend-registered) corpus. Fixture corpora keep whatever
  // data.ts ships.
  useEffect(() => {
    if (subview !== "coll" || !activeCorpus || !term.trim()) return;
    if (!inTauri() || !activeCorpus.id.startsWith("corpus-")) {
      setCollocates(null);
      return;
    }
    let cancelled = false;
    runCollocates({
      corpusId: activeCorpus.id,
      term: term.trim(),
      layer,
      window: 5,
      limit: 60,
    })
      .then((r) => {
        if (!cancelled) setCollocates(r.collocates);
      })
      .catch((e) => {
        console.error("runCollocates failed:", e);
        if (!cancelled) setCollocates([]);
      });
    return () => {
      cancelled = true;
    };
  }, [subview, activeCorpus, term, layer]);

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
    window.setTimeout(run, 20);
  };

  const onRunCmd = (cmd: CommandDef) => {
    if (cmd.id === "build") setBuildOpen(true);
    else if (cmd.id === "open") alert("Tauri open-dialog — wired in the real app");
    else if (cmd.id === "detail") setView("corpus");
    else if (cmd.id === "layer-word") {
      setLayer("word");
      window.setTimeout(run, 30);
    } else if (cmd.id === "layer-lemma") {
      if (activeCorpus?.annotated) {
        setLayer("lemma");
        setTerm("go");
        window.setTimeout(run, 30);
      }
    } else if (cmd.id === "layer-pos") {
      if (activeCorpus?.annotated) {
        setLayer("pos");
        setTerm("NN");
        window.setTimeout(run, 30);
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
            onOpen={() => {
              /* TODO: Tauri dialog */
            }}
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
          onLayer={(l) => {
            setLayer(l);
            window.setTimeout(run, 30);
          }}
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
                onSort={setSortMode}
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
            <CollocationsView corpus={activeCorpus} term={term} data={collocates} />
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
            window.setTimeout(run, 30);
          }}
          onOpen={() => alert("Tauri open-dialog — wired in the real app")}
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
