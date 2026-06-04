import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { QueryBar, type QueryBarProps } from "./QueryBar";

function setup(overrides: Partial<QueryBarProps> = {}) {
  const onFilterChange = vi.fn();
  const props: QueryBarProps = {
    layer: "word",
    term: "linguistic",
    onLayer: vi.fn(),
    onTerm: vi.fn(),
    onRun: vi.fn(),
    onOpenPalette: vi.fn(),
    filter: {},
    onFilterChange,
    filterable: true,
    ...overrides,
  };
  render(<QueryBar {...props} />);
  return { onFilterChange };
}

describe("QueryBar filter", () => {
  it("hides the filter control entirely when the corpus isn't filterable", () => {
    setup({ filterable: false, filter: { author: "x" } });
    expect(screen.queryByRole("button", { name: /filter/i })).toBeNull();
    // Not even an existing filter's chip leaks through on a fixture corpus.
    expect(screen.queryByText(/author: x/i)).toBeNull();
  });

  it("renders a removable chip per active dimension", () => {
    const { onFilterChange } = setup({ filter: { yearMin: 1800, yearMax: 1950 } });
    const chip = screen.getByText(/year: 1800–1950/i);
    expect(chip).toBeInTheDocument();
    fireEvent.click(chip);
    expect(onFilterChange).toHaveBeenCalledWith({}); // year cleared
  });

  it("flags a CQL query and mutes the layer toggle", () => {
    const { container } = render(
      <QueryBar
        layer="word"
        term={'[pos="NN"]'}
        onLayer={vi.fn()}
        onTerm={vi.fn()}
        onRun={vi.fn()}
        onOpenPalette={vi.fn()}
        filter={{}}
        onFilterChange={vi.fn()}
        filterable={false}
      />,
    );
    expect(screen.getByText("CQL")).toBeInTheDocument();
    expect(container.querySelector(".cx-layer-toggle.is-muted")).not.toBeNull();
  });

  it("shows the layer hint for a bare term, not CQL", () => {
    render(
      <QueryBar
        layer="word"
        term="bank"
        onLayer={vi.fn()}
        onTerm={vi.fn()}
        onRun={vi.fn()}
        onOpenPalette={vi.fn()}
        filter={{}}
        onFilterChange={vi.fn()}
        filterable={false}
      />,
    );
    expect(screen.getByText("regex ok")).toBeInTheDocument();
    expect(screen.queryByText("CQL")).toBeNull();
  });

  it("opens the popover and commits a normalized filter on Apply", () => {
    const { onFilterChange } = setup();
    fireEvent.click(screen.getByRole("button", { name: /filter/i }));
    fireEvent.change(screen.getByPlaceholderText("from"), { target: { value: "1850" } });
    fireEvent.change(screen.getByPlaceholderText("to"), { target: { value: "1900" } });
    fireEvent.change(screen.getByLabelText("author"), { target: { value: "  Austen  " } });
    fireEvent.click(screen.getByRole("button", { name: /apply/i }));
    expect(onFilterChange).toHaveBeenCalledWith({ yearMin: 1850, yearMax: 1900, author: "Austen" });
  });
});
