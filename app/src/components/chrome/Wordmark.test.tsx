import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Wordmark } from "./Wordmark";

describe("Wordmark", () => {
  it("renders the brand text", () => {
    render(<Wordmark />);
    expect(screen.getByText(/corpust/)).toBeInTheDocument();
  });

  it.each([
    ["xs", 13],
    ["sm", 22],
    ["md", 26],
    ["lg", 44],
  ] as const)("applies %s size as %dpx font", (size, px) => {
    const { container } = render(<Wordmark size={size} />);
    const el = container.firstChild as HTMLElement;
    expect(el.style.fontSize).toBe(`${px}px`);
  });
});
