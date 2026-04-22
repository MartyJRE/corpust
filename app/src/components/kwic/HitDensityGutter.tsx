// Novel surface: vertical minimap of hit density across the corpus. Each
// cell = bucketed slice of the doc-position axis; fill opacity ~ density.
// Click = jump scroll (wire to the KWIC column scrollTop).

export interface HitDensityGutterProps {
  density: number[];
  scrollPct: number;
  onJump: (pct: number) => void;
}

export function HitDensityGutter({ density, scrollPct, onJump }: HitDensityGutterProps) {
  const max = Math.max(1, ...density);
  return (
    <div className="cx-density-gutter" title="hit density across corpus position">
      {density.map((v, i) => {
        const op = 0.08 + 0.92 * (v / max);
        return (
          <div
            key={i}
            className="cx-density-cell"
            style={{
              background: `color-mix(in oklch, var(--accent) ${Math.round(op * 100)}%, transparent)`,
            }}
            onClick={() => onJump(i / density.length)}
            title={`${v} hit${v === 1 ? "" : "s"} · position ${Math.round((i / density.length) * 100)}%`}
          />
        );
      })}
      <div
        className="cx-density-thumb"
        style={{
          top: `calc(${scrollPct * 100}% * 0.9 + 6px)`,
          height: 14,
        }}
      />
    </div>
  );
}
