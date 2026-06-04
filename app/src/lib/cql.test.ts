import { describe, expect, it } from "vitest";
import { isCql } from "./cql";

describe("isCql", () => {
  it("treats bracket and quote forms as CQL", () => {
    expect(isCql('[pos="NN"]')).toBe(true);
    expect(isCql('"bank"')).toBe(true);
    expect(isCql('[word="a"] [word="b"]')).toBe(true);
  });
  it("leaves bare terms (incl. bare regex) as classic queries", () => {
    expect(isCql("bank")).toBe(false);
    expect(isCql("run.*")).toBe(false);
    expect(isCql("")).toBe(false);
  });
});
