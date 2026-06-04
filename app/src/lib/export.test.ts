import { describe, expect, it } from "vitest";
import { concordanceCsv, concordanceJson, documentsCsv, slug } from "./export";
import type { KwicResult } from "@/types";

const result: KwicResult = {
  hits: [
    { docId: "a.txt", pos: 10, left: "to be, or not", hit: "to", right: 'be — "that" is', lemma: "be", pos_tag: "VB" },
    { docId: "b.txt", pos: 22, left: "plain left", hit: "to", right: "plain right" },
  ],
  elapsedMs: 1.2,
  truncated: false,
  total: 2,
  offset: 0,
};

describe("concordanceCsv", () => {
  it("emits a header and one row per hit", () => {
    const lines = concordanceCsv(result).trimEnd().split("\r\n");
    expect(lines[0]).toBe("n,doc,position,left,node,right");
    expect(lines).toHaveLength(3);
    expect(lines[2]).toBe("2,b.txt,22,plain left,to,plain right");
  });

  it("quotes fields containing commas, quotes, or newlines and doubles inner quotes", () => {
    const row = concordanceCsv(result).split("\r\n")[1];
    // left "to be, or not" → quoted (comma); right with embedded quotes → doubled
    expect(row).toContain('"to be, or not"');
    expect(row).toContain('"be — ""that"" is"');
  });
});

describe("concordanceJson", () => {
  it("carries query context and per-hit fields, omitting absent lemma/pos", () => {
    const parsed = JSON.parse(concordanceJson(result, null, "to", "word"));
    expect(parsed).toMatchObject({ corpus: null, term: "to", layer: "word", total: 2 });
    expect(parsed.hits[0]).toMatchObject({ doc: "a.txt", node: "to", lemma: "be", posTag: "VB" });
    expect(parsed.hits[1]).not.toHaveProperty("lemma");
    expect(parsed.hits[1]).not.toHaveProperty("posTag");
  });
});

describe("documentsCsv", () => {
  it("renders null title/author/year as empty cells", () => {
    const csv = documentsCsv([{ file: "a.txt", title: null, author: null, year: null, tokens: 99 }]);
    const lines = csv.trimEnd().split("\r\n");
    expect(lines[0]).toBe("file,title,author,year,tokens");
    expect(lines[1]).toBe("a.txt,,,,99");
  });
});

describe("slug", () => {
  it("lowercases, collapses non-alphanumerics, and trims dashes", () => {
    expect(slug("Gutenberg · EN")).toBe("gutenberg-en");
    expect(slug("  NYT 2020–2025  ")).toBe("nyt-2020-2025");
  });

  it("falls back to 'corpus' for empty input", () => {
    expect(slug("···")).toBe("corpus");
  });
});
