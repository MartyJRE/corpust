export function Wordmark({ size = "sm" }: { size?: "sm" | "md" | "lg" }) {
  const fs = size === "lg" ? 44 : size === "md" ? 26 : 22;
  return (
    <div
      style={{
        fontFamily: "var(--font-serif)",
        fontSize: fs,
        letterSpacing: "-0.015em",
        lineHeight: 1,
        color: "var(--fg)",
      }}
    >
      corpust<span style={{ color: "var(--accent)" }}>.</span>
    </div>
  );
}
