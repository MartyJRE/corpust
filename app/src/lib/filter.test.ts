import { describe, expect, it } from "vitest";
import {
  clearDimension,
  docMatchesFilter,
  filterChips,
  isFilterEmpty,
  normalizeFilter,
} from "./filter";

describe("isFilterEmpty", () => {
  it("is true for {} and whitespace-only dimensions", () => {
    expect(isFilterEmpty({})).toBe(true);
    expect(isFilterEmpty({ author: "  ", path: "" })).toBe(true);
  });
  it("is false once any dimension is set", () => {
    expect(isFilterEmpty({ yearMin: 1900 })).toBe(false);
    expect(isFilterEmpty({ author: "twain" })).toBe(false);
  });
});

describe("normalizeFilter", () => {
  it("trims, drops empties, and returns undefined when nothing remains", () => {
    expect(normalizeFilter({ author: "  ", path: "" })).toBeUndefined();
    expect(normalizeFilter({ author: "  Twain  ", yearMin: 1880 })).toEqual({
      author: "Twain",
      yearMin: 1880,
    });
  });
});

describe("filterChips", () => {
  it("renders one chip per active dimension with open-ended year bounds", () => {
    expect(filterChips({ yearMin: 1800, yearMax: 1950 })).toEqual([
      { key: "year", label: "year: 1800–1950" },
    ]);
    expect(filterChips({ yearMin: 1800 })[0].label).toBe("year: 1800–…");
    expect(filterChips({ author: "Austen", path: "fiction" }).map((c) => c.key)).toEqual([
      "author",
      "path",
    ]);
  });
});

describe("clearDimension", () => {
  it("removes both year bounds together, others individually", () => {
    expect(clearDimension({ yearMin: 1, yearMax: 2, author: "x" }, "year")).toEqual({ author: "x" });
    expect(clearDimension({ author: "x", path: "y" }, "author")).toEqual({ path: "y" });
  });
});

describe("docMatchesFilter", () => {
  const doc = { path: "corpus/fiction/austen-pp.txt", author: "Jane Austen", year: 1813 };
  it("matches everything with no filter", () => {
    expect(docMatchesFilter(doc, undefined)).toBe(true);
  });
  it("applies year range and excludes undated docs", () => {
    expect(docMatchesFilter(doc, { yearMin: 1800, yearMax: 1820 })).toBe(true);
    expect(docMatchesFilter(doc, { yearMin: 1900 })).toBe(false);
    expect(docMatchesFilter({ ...doc, year: null }, { yearMin: 1800 })).toBe(false);
  });
  it("matches author and path case-insensitively as substrings", () => {
    expect(docMatchesFilter(doc, { author: "austen" })).toBe(true);
    expect(docMatchesFilter(doc, { author: "dickens" })).toBe(false);
    expect(docMatchesFilter(doc, { path: "FICTION" })).toBe(true);
    expect(docMatchesFilter(doc, { path: "legal" })).toBe(false);
  });
});
