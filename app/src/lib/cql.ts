// Whether a query string should be treated as CQL (token-attribute /
// multi-token) rather than a bare single-layer term. Mirrors the backend
// `corpust_query::is_cql` so the UI and engine agree on what's a CQL query.

export function isCql(input: string): boolean {
  const t = input.trimStart();
  return t.startsWith("[") || t.startsWith('"') || input.includes("[");
}
