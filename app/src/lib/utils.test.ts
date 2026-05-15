import { describe, expect, it } from "vitest";
import {
  cn,
  formatBuildTime,
  formatBytes,
  formatDate,
  formatDuration,
  formatNumber,
  makeDensity,
} from "./utils";

describe("cn", () => {
  it("concatenates truthy class names", () => {
    expect(cn("a", "b")).toContain("a");
    expect(cn("a", "b")).toContain("b");
  });

  it("drops falsy entries", () => {
    expect(cn("a", false && "b", null, undefined, "c")).toBe("a c");
  });

  it("merges tailwind utilities (last wins on conflict)", () => {
    expect(cn("px-2", "px-4")).toBe("px-4");
  });
});

describe("formatBytes", () => {
  it("returns bytes under 1 KiB", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1023)).toBe("1023 B");
  });

  it("scales to KB / MB / GB", () => {
    expect(formatBytes(1024)).toBe("1.0 KB");
    expect(formatBytes(1024 * 1024)).toBe("1.0 MB");
    expect(formatBytes(1024 ** 3)).toBe("1.0 GB");
  });

  it("drops decimal once values exceed 10 of a unit", () => {
    expect(formatBytes(12 * 1024)).toBe("12 KB");
  });

  it("caps at TB", () => {
    expect(formatBytes(1024 ** 4)).toMatch(/TB$/);
  });
});

describe("formatDuration", () => {
  it("uses µs under 1 ms", () => {
    expect(formatDuration(0.5)).toBe("500 µs");
  });

  it("uses ms under 1 s", () => {
    expect(formatDuration(42.3)).toBe("42.3 ms");
  });

  it("uses s once over 1 s", () => {
    expect(formatDuration(1500)).toBe("1.50 s");
  });
});

describe("formatDate", () => {
  it("renders an ISO date in en-GB", () => {
    expect(formatDate("2025-03-15T12:00:00Z")).toMatch(/Mar/);
    expect(formatDate("2025-03-15T12:00:00Z")).toMatch(/2025/);
  });
});

describe("formatBuildTime", () => {
  it("uses seconds under a minute", () => {
    expect(formatBuildTime(15_000)).toBe("15.0 s");
  });

  it("switches to m:ss past a minute", () => {
    expect(formatBuildTime(95_000)).toBe("1:35");
  });
});

describe("formatNumber", () => {
  it("groups thousands", () => {
    expect(formatNumber(1234567)).toBe("1,234,567");
  });
});

describe("makeDensity", () => {
  it("returns an empty-shaped array for no hits", () => {
    expect(makeDensity([], 10)).toHaveLength(10);
  });

  it("buckets hits by position", () => {
    const hits = Array.from({ length: 5 }, (_, i) => ({ pos: i * 10 }));
    const d = makeDensity(hits, 5);
    expect(d).toHaveLength(5);
    const sum = d.reduce((a, b) => a + b, 0);
    // Each hit lands in exactly one bucket; smearing may add 0/1s
    // outside the source set, but the real hits never get lost.
    expect(sum).toBeGreaterThanOrEqual(hits.length);
  });
});
