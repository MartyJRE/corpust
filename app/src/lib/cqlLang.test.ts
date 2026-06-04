import { describe, expect, it } from "vitest";
import { completionAt, tokenizeCql, validateCql } from "./cqlLang";

describe("tokenizeCql", () => {
  it("returns nothing for a bare (non-CQL) term", () => {
    expect(tokenizeCql("linguistic")).toEqual([]);
    expect(tokenizeCql("run.*")).toEqual([]);
  });

  it("classifies brackets, attributes, operators, and values", () => {
    const kinds = tokenizeCql('[word="bank" pos="N.*"]').map((t) => t.kind);
    expect(kinds).toEqual(["bracket", "attr", "operator", "string", "attr", "operator", "regex", "bracket"]);
  });

  it("marks unknown identifiers and stray chars invalid", () => {
    const toks = tokenizeCql('[foo="x"]');
    expect(toks.find((t) => t.kind === "attr")).toBeUndefined();
    expect(toks.some((t) => t.kind === "invalid")).toBe(true);
  });
});

describe("validateCql", () => {
  it("accepts valid queries (and bare terms)", () => {
    expect(validateCql("linguistic")).toBeNull();
    expect(validateCql('[word="bank" pos="NN"]')).toBeNull();
    expect(validateCql('"the" [pos="N.*"]')).toBeNull();
  });

  it("flags each malformed shape with a span + message", () => {
    expect(validateCql('[word="a"')?.message).toMatch(/missing `\]`/);
    expect(validateCql("[]")?.message).toMatch(/empty token/);
    expect(validateCql('[foo="x"]')?.message).toMatch(/unknown attribute/);
    expect(validateCql('[word"x"]')?.message).toMatch(/expected `=`/);
    expect(validateCql('[word=x]')?.message).toMatch(/expected a quoted value/);
    expect(validateCql('[word="("]')?.message).toMatch(/invalid regex/);
    const d = validateCql("[]");
    expect(d && d.to > d.from).toBe(true);
  });
});

describe("completionAt", () => {
  it("offers attribute completion inside an empty bracket", () => {
    const q = "[";
    expect(completionAt(q, q.length)).toEqual({ kind: "attr", from: 1 });
  });

  it("offers POS-value completion inside a pos/tag value", () => {
    const q = '[pos="N';
    expect(completionAt(q, q.length).kind).toBe("posValue");
    const q2 = '[tag="';
    expect(completionAt(q2, q2.length).kind).toBe("posValue");
  });

  it("offers nothing inside a word value or outside brackets", () => {
    const q = '[word="ba';
    expect(completionAt(q, q.length).kind).toBe("none");
    expect(completionAt("bank", 4).kind).toBe("none");
    expect(completionAt('[a="x"] ', 8).kind).toBe("none");
  });
});
