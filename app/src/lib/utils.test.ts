import { describe, expect, it } from "vitest";
import { formatBytes, formatDuration, formatNumber } from "./utils";

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

describe("formatNumber", () => {
  it("groups thousands", () => {
    expect(formatNumber(1234567)).toBe("1,234,567");
  });
});
