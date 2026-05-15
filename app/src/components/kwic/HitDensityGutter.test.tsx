import { describe, expect, it, vi } from "vitest";
import { fireEvent, render } from "@testing-library/react";
import { HitDensityGutter } from "./HitDensityGutter";

describe("HitDensityGutter", () => {
  it("renders one cell per density bucket", () => {
    const { container } = render(
      <HitDensityGutter density={[0, 1, 2]} scrollPct={0} onJump={() => {}} />,
    );
    expect(container.querySelectorAll(".cx-density-cell")).toHaveLength(3);
  });

  it("calls onJump with the clicked cell fraction", () => {
    const onJump = vi.fn();
    const { container } = render(
      <HitDensityGutter
        density={[0, 1, 2, 3]}
        scrollPct={0}
        onJump={onJump}
      />,
    );
    const cells = container.querySelectorAll(".cx-density-cell");
    fireEvent.click(cells[2]);
    expect(onJump).toHaveBeenCalledWith(2 / 4);
  });

  it("positions the scroll thumb proportionally", () => {
    const { container } = render(
      <HitDensityGutter density={[1, 2]} scrollPct={0.5} onJump={() => {}} />,
    );
    const thumb = container.querySelector(".cx-density-thumb") as HTMLElement;
    // jsdom's CSSOM evaluates `calc(50% * 0.9 + 6px)` to "calc(45% + 6px)".
    expect(thumb.style.top).toBe("calc(45% + 6px)");
  });

  it("anchors the scroll thumb at zero when scrollPct is 0", () => {
    const { container } = render(
      <HitDensityGutter density={[1, 2]} scrollPct={0} onJump={() => {}} />,
    );
    const thumb = container.querySelector(".cx-density-thumb") as HTMLElement;
    expect(thumb.style.top).toBe("calc(0% + 6px)");
  });
});
