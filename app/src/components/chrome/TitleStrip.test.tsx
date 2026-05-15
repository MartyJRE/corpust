import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { TitleStrip } from "./TitleStrip";

describe("TitleStrip", () => {
  it("renders the three main tabs", () => {
    render(<TitleStrip view="search" onView={() => {}} />);
    expect(screen.getByRole("button", { name: /search/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /corpus/i })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /settings/i }),
    ).toBeInTheDocument();
  });

  it("marks the active tab with is-on", () => {
    render(<TitleStrip view="corpus" onView={() => {}} />);
    const corpus = screen.getByRole("button", { name: /corpus/i });
    expect(corpus.className).toContain("is-on");
    const search = screen.getByRole("button", { name: /search/i });
    expect(search.className).not.toContain("is-on");
  });

  it("calls onView with the clicked tab id", () => {
    const onView = vi.fn();
    render(<TitleStrip view="search" onView={onView} />);
    fireEvent.click(screen.getByRole("button", { name: /settings/i }));
    expect(onView).toHaveBeenCalledWith("settings");
  });
});
