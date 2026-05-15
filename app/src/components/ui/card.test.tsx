import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Card, CardContent, CardHeader, CardTitle } from "./card";

describe("Card", () => {
  it("composes nested header/title/content", () => {
    render(
      <Card data-testid="card">
        <CardHeader>
          <CardTitle>title</CardTitle>
        </CardHeader>
        <CardContent>body</CardContent>
      </Card>,
    );
    expect(screen.getByTestId("card")).toBeInTheDocument();
    expect(screen.getByText("title")).toBeInTheDocument();
    expect(screen.getByText("body")).toBeInTheDocument();
  });

  it("forwards className through cn()", () => {
    render(<Card className="custom-class">x</Card>);
    expect(screen.getByText("x")).toHaveClass("custom-class");
    // base class survives.
    expect(screen.getByText("x").className).toContain("rounded-lg");
  });
});
