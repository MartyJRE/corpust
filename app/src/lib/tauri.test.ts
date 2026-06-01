import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const invokeMock = vi.fn();

vi.mock("@tauri-apps/api/core", () => ({
  invoke: (...args: unknown[]) => invokeMock(...args),
}));

describe("tauri wrappers", () => {
  beforeEach(() => {
    invokeMock.mockReset();
  });

  afterEach(() => {
    // Clean up any window mutations the inTauri test might have made.
    delete (window as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
  });

  it("listCorpora invokes the list_corpora command", async () => {
    invokeMock.mockResolvedValue([]);
    const { listCorpora } = await import("./tauri");
    const out = await listCorpora();
    expect(invokeMock).toHaveBeenCalledWith("list_corpora", undefined);
    expect(out).toEqual([]);
  });

  it("runKwic forwards the request under `req`", async () => {
    invokeMock.mockResolvedValue({ hits: [], elapsedMs: 0 });
    const { runKwic } = await import("./tauri");
    const req = {
      corpusId: "c1",
      term: "the",
      layer: "word" as const,
      context: 5,
      limit: 50,
    };
    await runKwic(req);
    expect(invokeMock).toHaveBeenCalledWith("run_kwic", { req });
  });

  it("runCollocates forwards the request under `req`", async () => {
    invokeMock.mockResolvedValue({
      collocates: [],
      elapsedMs: 1,
      nodeHits: 0,
      windowTokens: 0,
    });
    const { runCollocates } = await import("./tauri");
    const req = {
      corpusId: "c1",
      term: "the",
      layer: "word" as const,
      leftWindow: 5,
      rightWindow: 5,
      limit: 25,
    };
    await runCollocates(req);
    expect(invokeMock).toHaveBeenCalledWith("run_collocates", { req });
  });

  it("buildIndex forwards the request under `req`", async () => {
    invokeMock.mockResolvedValue({});
    const { buildIndex } = await import("./tauri");
    const req = { name: "x", sources: [] } as unknown as Parameters<
      typeof buildIndex
    >[0];
    await buildIndex(req);
    expect(invokeMock).toHaveBeenCalledWith("build_index", { req });
  });

  it("listDocuments forwards corpusId", async () => {
    invokeMock.mockResolvedValue([]);
    const { listDocuments } = await import("./tauri");
    await listDocuments("c1");
    expect(invokeMock).toHaveBeenCalledWith("list_documents", { corpusId: "c1" });
  });

  it("listDocuments carries title/author/year through (incl. nulls)", async () => {
    invokeMock.mockResolvedValue([
      {
        docId: 0,
        path: "84.txt",
        tokenCount: 75000,
        title: "Frankenstein",
        author: "Mary Shelley",
        year: 1818,
      },
      {
        docId: 1,
        path: "unknown.txt",
        tokenCount: 100,
        title: null,
        author: null,
        year: null,
      },
    ]);
    const { listDocuments } = await import("./tauri");
    const docs = await listDocuments("c1");
    expect(docs[0]).toMatchObject({ title: "Frankenstein", author: "Mary Shelley", year: 1818 });
    expect(docs[1]).toMatchObject({ title: null, author: null, year: null });
  });

  it("runFrequencies forwards the request under `req`", async () => {
    invokeMock.mockResolvedValue({ rows: [], totalTokens: 0, elapsedMs: 0 });
    const { runFrequencies } = await import("./tauri");
    const req = { corpusId: "c1", layer: "word" as const, limit: 12 };
    await runFrequencies(req);
    expect(invokeMock).toHaveBeenCalledWith("run_frequencies", { req });
  });

  it("runTermDistribution forwards the request under `req`", async () => {
    invokeMock.mockResolvedValue({
      docCounts: [],
      dispersion: [],
      totalHits: 0,
      elapsedMs: 0,
    });
    const { runTermDistribution } = await import("./tauri");
    const req = { corpusId: "c1", term: "the", layer: "word" as const, buckets: 100 };
    await runTermDistribution(req);
    expect(invokeMock).toHaveBeenCalledWith("run_term_distribution", { req });
  });

  it("expandContext forwards the request under `req`", async () => {
    invokeMock.mockResolvedValue({
      docId: 0,
      path: "a.txt",
      before: "",
      hit: "the",
      after: "",
      tokenCount: 10,
    });
    const { expandContext } = await import("./tauri");
    const req = { corpusId: "c1", docId: 0, position: 5, context: 45 };
    await expandContext(req);
    expect(invokeMock).toHaveBeenCalledWith("expand_context", { req });
  });

  it("isFixtureCorpus recognises baked-in demo corpora", async () => {
    const { isFixtureCorpus } = await import("./tauri");
    expect(isFixtureCorpus("does-not-exist")).toBe(false);
    expect(isFixtureCorpus("gut-en")).toBe(true);
  });

  it("inTauri detects the runtime marker", async () => {
    const { inTauri } = await import("./tauri");
    expect(inTauri()).toBe(false);
    (window as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ = {};
    expect(inTauri()).toBe(true);
  });
});
