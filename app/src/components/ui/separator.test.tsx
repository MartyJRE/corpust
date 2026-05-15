import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import { Separator } from "./separator";

describe("Separator", () => {
  it("defaults to horizontal", () => {
    const { container } = render(<Separator />);
    const el = container.firstChild as HTMLElement;
    expect(el.getAttribute("data-orientation")).toBe("horizontal");
    expect(el.className).toContain("h-px");
  });

  it("switches to vertical when asked", () => {
    const { container } = render(<Separator orientation="vertical" />);
    const el = container.firstChild as HTMLElement;
    expect(el.getAttribute("data-orientation")).toBe("vertical");
    expect(el.className).toContain("w-px");
  });
});
