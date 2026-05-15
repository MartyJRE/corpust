import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { Input } from "./input";

describe("Input", () => {
  it("renders as a native input", () => {
    render(<Input placeholder="search" />);
    expect(screen.getByPlaceholderText("search").tagName).toBe("INPUT");
  });

  it("supports controlled value changes", () => {
    const handler = vi.fn();
    render(<Input onChange={handler} placeholder="x" />);
    fireEvent.change(screen.getByPlaceholderText("x"), {
      target: { value: "hi" },
    });
    expect(handler).toHaveBeenCalledOnce();
  });

  it("forwards type attribute", () => {
    render(<Input type="number" placeholder="n" />);
    expect(screen.getByPlaceholderText("n")).toHaveAttribute("type", "number");
  });
});
