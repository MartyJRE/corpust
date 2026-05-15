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

  it("inTauri detects the runtime marker", async () => {
    const { inTauri } = await import("./tauri");
    expect(inTauri()).toBe(false);
    (window as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ = {};
    expect(inTauri()).toBe(true);
  });
});
