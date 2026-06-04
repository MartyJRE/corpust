// CQL language support for the query editor: a tokenizer (syntax
// highlighting), a validator (inline error squiggles), and completion
// context. Framework-agnostic and unit-tested; the CodeMirror glue lives
// in CqlInput.tsx. Mirrors the Rust grammar/errors in
// crates/corpust-query/src/lib.rs — keep the two in sync.

import { isCql } from "./cql";

/** Known attributes and their canonical layer (aliases included). */
export const ATTRS = ["word", "lemma", "pos", "hw", "tag"] as const;

/** Common Penn-Treebank-ish POS tags, for value completion on pos/tag.
 *  (Corpus-derived tags are a later enhancement.) */
export const POS_TAGS = [
  "NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
  "JJ", "JJR", "JJS", "RB", "RBR", "RBS", "DT", "IN", "CC", "CD",
  "PRP", "PRP$", "WP", "WDT", "MD", "TO", "UH", "FW",
] as const;

/** A completion candidate with JetBrains-style detail + doc. */
export interface CqlCompletion {
  label: string;
  /** Short type shown right-aligned (e.g. "layer", "POS"). */
  detail: string;
  /** Longer description shown in the side panel. */
  info: string;
}

/** Attribute completions (with aliases), richest first. */
export const ATTR_COMPLETIONS: CqlCompletion[] = [
  { label: "word", detail: "layer", info: "Surface form of the token (case-insensitive)." },
  { label: "lemma", detail: "layer", info: "Dictionary form. Requires an annotated corpus." },
  { label: "pos", detail: "layer", info: "Part-of-speech tag, e.g. NN, VBD (case-sensitive)." },
  { label: "hw", detail: "alias → lemma", info: "Headword — alias for lemma." },
  { label: "tag", detail: "alias → pos", info: "Alias for pos." },
];

/** Penn-Treebank tag → human label, for POS value completion + docs. */
export const POS_LABELS: Record<string, string> = {
  NN: "noun, singular", NNS: "noun, plural", NNP: "proper noun, singular",
  NNPS: "proper noun, plural", VB: "verb, base form", VBD: "verb, past tense",
  VBG: "verb, gerund/present participle", VBN: "verb, past participle",
  VBP: "verb, non-3rd person singular present", VBZ: "verb, 3rd person singular present",
  JJ: "adjective", JJR: "adjective, comparative", JJS: "adjective, superlative",
  RB: "adverb", RBR: "adverb, comparative", RBS: "adverb, superlative",
  DT: "determiner", IN: "preposition / subordinating conjunction",
  CC: "coordinating conjunction", CD: "cardinal number", PRP: "personal pronoun",
  "PRP$": "possessive pronoun", WP: "wh-pronoun", WDT: "wh-determiner",
  MD: "modal", TO: "to", UH: "interjection", FW: "foreign word",
};

export const POS_COMPLETIONS: CqlCompletion[] = POS_TAGS.map((t) => ({
  label: t,
  detail: "POS",
  info: POS_LABELS[t] ?? "part-of-speech tag",
}));

export type TokenKind = "bracket" | "attr" | "operator" | "string" | "regex" | "invalid";

export interface CqlToken {
  from: number;
  to: number;
  kind: TokenKind;
}

export interface CqlDiagnostic {
  from: number;
  to: number;
  message: string;
}

const ATTR_SET: ReadonlySet<string> = new Set(ATTRS);
const REGEX_META = /[.*+?^${}()|[\]\\]/;

function isIdentChar(c: string): boolean {
  return /[A-Za-z0-9_]/.test(c);
}

/** Classify a quoted value as a regex (has metachars) or a plain string. */
function valueKind(value: string): TokenKind {
  return REGEX_META.test(value) ? "regex" : "string";
}

/** Tokenize for highlighting. Returns marks in document order. Bare
 *  (non-CQL) input yields no tokens so it renders as plain text. */
export function tokenizeCql(text: string): CqlToken[] {
  if (!isCql(text)) return [];
  const tokens: CqlToken[] = [];
  let i = 0;
  let depth = 0;
  while (i < text.length) {
    const c = text[i];
    if (/\s/.test(c)) {
      i++;
      continue;
    }
    if (c === "[") {
      tokens.push({ from: i, to: i + 1, kind: "bracket" });
      depth++;
      i++;
    } else if (c === "]") {
      tokens.push({ from: i, to: i + 1, kind: "bracket" });
      depth = Math.max(0, depth - 1);
      i++;
    } else if (c === "=") {
      tokens.push({ from: i, to: i + 1, kind: depth > 0 ? "operator" : "invalid" });
      i++;
    } else if (c === '"') {
      const start = i;
      i++;
      while (i < text.length && text[i] !== '"') i++;
      const closed = i < text.length;
      const value = text.slice(start + 1, i);
      if (closed) i++; // consume closing quote
      tokens.push({ from: start, to: i, kind: closed ? valueKind(value) : "invalid" });
    } else if (isIdentChar(c)) {
      const start = i;
      while (i < text.length && isIdentChar(text[i])) i++;
      const ident = text.slice(start, i);
      const kind: TokenKind = depth > 0 && ATTR_SET.has(ident) ? "attr" : "invalid";
      tokens.push({ from: start, to: i, kind });
    } else {
      tokens.push({ from: i, to: i + 1, kind: "invalid" });
      i++;
    }
  }
  return tokens;
}

/** Validate a CQL string, returning the first error (or `null` when valid
 *  / not a CQL query). Mirrors `parse_cql` in corpust-query. */
export function validateCql(text: string): CqlDiagnostic | null {
  if (!isCql(text)) return null;

  let i = 0;
  const n = text.length;
  const skipWs = () => {
    while (i < n && /\s/.test(text[i])) i++;
  };
  let tokenCount = 0;

  skipWs();
  while (i < n) {
    const c = text[i];
    if (c === "[") {
      const err = parseBracket();
      if (err) return err;
      tokenCount++;
    } else if (c === '"') {
      const err = parseQuoted();
      if (err) return err;
      tokenCount++;
    } else {
      return { from: i, to: i + 1, message: `unexpected \`${c}\` — expected a \`[…]\` token or a quoted string` };
    }
    skipWs();
  }
  if (tokenCount === 0) return { from: 0, to: n, message: "empty query" };
  return null;

  // --- helpers (close over i/text) ---
  function parseQuoted(): CqlDiagnostic | null {
    const start = i;
    i++; // opening quote
    while (i < n && text[i] !== '"') i++;
    if (i >= n) return { from: start, to: n, message: "unterminated quoted value" };
    i++; // closing quote
    return checkRegex(text.slice(start + 1, i - 1), start, i);
  }

  function parseBracket(): CqlDiagnostic | null {
    const bracketStart = i;
    i++; // '['
    let constraints = 0;
    for (;;) {
      skipWs();
      if (i >= n) return { from: bracketStart, to: n, message: "unterminated `[` — missing `]`" };
      const c = text[i];
      if (c === "]") {
        i++;
        if (constraints === 0) {
          return { from: bracketStart, to: i, message: "empty token `[]` — give it at least one attribute" };
        }
        return null;
      }
      if (/[A-Za-z]/.test(c)) {
        const attrFrom = i;
        while (i < n && isIdentChar(text[i])) i++;
        const attr = text.slice(attrFrom, i);
        if (!ATTR_SET.has(attr)) {
          return { from: attrFrom, to: i, message: `unknown attribute \`${attr}\` — use word, lemma (hw), or pos (tag)` };
        }
        skipWs();
        if (text[i] !== "=") return { from: i, to: i + 1, message: `expected \`=\` after attribute \`${attr}\`` };
        i++; // '='
        skipWs();
        if (text[i] !== '"') return { from: i, to: i + 1, message: `expected a quoted value after \`${attr}=\`` };
        const err = parseQuoted();
        if (err) return err;
        constraints++;
      } else {
        return { from: i, to: i + 1, message: `unexpected \`${c}\` inside \`[…]\` — expected an attribute name or \`]\`` };
      }
    }
  }

  function checkRegex(value: string, from: number, to: number): CqlDiagnostic | null {
    if (value.length === 0) return { from, to, message: "empty value" };
    if (!REGEX_META.test(value)) return null; // plain value, exact match
    try {
      // Validate the same shape the backend compiles.
      void new RegExp(`^(?:${value})$`);
      return null;
    } catch (e) {
      return { from, to, message: `invalid regex \`${value}\`: ${(e as Error).message}` };
    }
  }
}

export type CompletionSpot =
  | { kind: "attr"; from: number }
  | { kind: "posValue"; from: number }
  | { kind: "none" };

/** What completion makes sense at `pos`: an attribute name inside a
 *  bracket, or a POS-tag value inside a `pos="…"` / `tag="…"`. */
export function completionAt(text: string, pos: number): CompletionSpot {
  const before = text.slice(0, pos);
  const lastOpen = before.lastIndexOf("[");
  const lastClose = before.lastIndexOf("]");
  if (lastOpen < 0 || lastOpen < lastClose) return { kind: "none" };
  const inBracket = before.slice(lastOpen);
  // An odd number of quotes since `[` means we're inside a value.
  const quotes = (inBracket.match(/"/g) ?? []).length;
  if (quotes % 2 === 1) {
    // Inside a value — offer POS tags only for pos/tag attributes.
    const attrMatch = inBracket.match(/([A-Za-z]+)\s*=\s*"[^"]*$/);
    const attr = attrMatch?.[1];
    if (attr === "pos" || attr === "tag") {
      const valStart = before.lastIndexOf('"') + 1;
      return { kind: "posValue", from: valStart };
    }
    return { kind: "none" };
  }
  // Outside a value → attribute-name position. Complete the current word.
  const wordMatch = before.match(/[A-Za-z]*$/);
  return { kind: "attr", from: pos - (wordMatch?.[0].length ?? 0) };
}
