import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { Button } from "./button";

describe("Button", () => {
  it("renders as a native button by default", () => {
    render(<Button>click</Button>);
    const btn = screen.getByRole("button", { name: "click" });
    expect(btn.tagName).toBe("BUTTON");
  });

  it("merges class names via cva", () => {
    render(<Button variant="outline">outlined</Button>);
    expect(screen.getByRole("button").className).toContain("border");
  });

  it("supports asChild to render arbitrary elements", () => {
    render(
      <Button asChild>
        <a href="#x">link</a>
      </Button>,
    );
    const link = screen.getByRole("link", { name: "link" });
    expect(link.tagName).toBe("A");
  });

  it("fires onClick when not disabled", () => {
    const handler = vi.fn();
    render(<Button onClick={handler}>go</Button>);
    fireEvent.click(screen.getByRole("button"));
    expect(handler).toHaveBeenCalledOnce();
  });

  it("respects disabled", () => {
    const handler = vi.fn();
    render(
      <Button onClick={handler} disabled>
        no
      </Button>,
    );
    fireEvent.click(screen.getByRole("button"));
    expect(handler).not.toHaveBeenCalled();
  });
});
