import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ViewTabs } from "./ViewTabs";
import type { KwicResult } from "@/types";

const result: KwicResult = {
  hits: [
    { docId: "d1", pos: 0, left: "", hit: "x", right: "" },
    { docId: "d1", pos: 1, left: "", hit: "x", right: "" },
  ],
  elapsedMs: 0,
  truncated: false,
};

describe("ViewTabs", () => {
  it("renders the three sub-views", () => {
    render(<ViewTabs view="kwic" onView={() => {}} result={null} />);
    expect(
      screen.getByRole("button", { name: /concordance/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /collocations/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /frequency/i }),
    ).toBeInTheDocument();
  });

  it("shows hit count on kwic tab when result is present", () => {
    render(<ViewTabs view="kwic" onView={() => {}} result={result} />);
    expect(
      screen.getByRole("button", { name: /concordance.*2/i }),
    ).toBeInTheDocument();
  });

  it("does not show a count on kwic when result is null", () => {
    render(<ViewTabs view="kwic" onView={() => {}} result={null} />);
    const btn = screen.getByRole("button", { name: /concordance/i });
    // count span only renders for non-null counts.
    expect(btn.querySelector(".count")).toBeNull();
  });

  it("calls onView with the clicked sub-view", () => {
    const onView = vi.fn();
    render(<ViewTabs view="kwic" onView={onView} result={null} />);
    fireEvent.click(screen.getByRole("button", { name: /collocations/i }));
    expect(onView).toHaveBeenCalledWith("coll");
  });
});
