// A single-line CodeMirror 6 editor for the query box: CQL syntax
// highlighting, bracket matching/autoclose, attribute/POS completion, and
// inline error squiggles. Honors the same contract as the plain input it
// replaces — value / onChange / onRun (Enter) / disabled / placeholder —
// and accepts bare terms unchanged. Colors come from app CSS variables
// (see .cm-cql-* in index.css) so it follows any future theme.

import {
  type CompletionContext,
  type CompletionResult,
  autocompletion,
  closeBrackets,
  closeBracketsKeymap,
  completionKeymap,
} from "@codemirror/autocomplete";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { bracketMatching } from "@codemirror/language";
import { linter } from "@codemirror/lint";
import { Compartment, EditorState, RangeSetBuilder } from "@codemirror/state";
import {
  Decoration,
  type DecorationSet,
  EditorView,
  ViewPlugin,
  type ViewUpdate,
  keymap,
  placeholder as cmPlaceholder,
} from "@codemirror/view";
import { useEffect, useRef } from "react";
import {
  ATTR_COMPLETIONS,
  POS_COMPLETIONS,
  completionAt,
  tokenizeCql,
  validateCql,
} from "@/lib/cqlLang";

export interface CqlInputProps {
  value: string;
  onChange: (value: string) => void;
  onRun: () => void;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

/** Decoration plugin: re-mark CQL tokens on every doc change. */
const cqlHighlight = ViewPlugin.fromClass(
  class {
    decorations: DecorationSet;
    constructor(view: EditorView) {
      this.decorations = buildDecorations(view);
    }
    update(u: ViewUpdate) {
      if (u.docChanged || u.viewportChanged) this.decorations = buildDecorations(u.view);
    }
  },
  { decorations: (v) => v.decorations },
);

function buildDecorations(view: EditorView): DecorationSet {
  const builder = new RangeSetBuilder<Decoration>();
  for (const tok of tokenizeCql(view.state.doc.toString())) {
    builder.add(tok.from, tok.to, Decoration.mark({ class: `cm-cql-${tok.kind}` }));
  }
  return builder.finish();
}

const cqlLinter = linter((view) => {
  const d = validateCql(view.state.doc.toString());
  if (!d) return [];
  return [{ from: d.from, to: Math.max(d.to, d.from + 1), severity: "error" as const, message: d.message }];
});

function cqlCompletion(ctx: CompletionContext): CompletionResult | null {
  const spot = completionAt(ctx.state.doc.toString(), ctx.pos);
  if (spot.kind === "attr") {
    return {
      from: spot.from,
      options: ATTR_COMPLETIONS.map((c) => ({
        label: c.label,
        detail: c.detail,
        info: c.info,
        type: "property",
      })),
      validFor: /^[A-Za-z]*$/,
    };
  }
  if (spot.kind === "posValue") {
    return {
      from: spot.from,
      options: POS_COMPLETIONS.map((c) => ({
        label: c.label,
        detail: c.detail,
        info: c.info,
        type: "enum",
      })),
      validFor: /^[A-Za-z$]*$/,
    };
  }
  return null;
}

/** Keep the editor to one line: cancel any transaction that introduces a
 *  newline (e.g. paste of multi-line text collapses to nothing extra). */
const singleLine = EditorState.transactionFilter.of((tr) => (tr.newDoc.lines > 1 ? [] : tr));

/** Editor chrome matched to `.cx-input` (34px, inset bg, accent focus). */
const theme = EditorView.theme({
  "&": {
    flex: "1",
    minWidth: "0",
    height: "34px",
    background: "var(--bg-inset)",
    border: "1px solid var(--border)",
    borderRadius: "5px",
    color: "var(--fg)",
    fontSize: "13px",
  },
  "&.cm-focused": {
    outline: "none",
    borderColor: "var(--accent)",
    boxShadow: "0 0 0 2px color-mix(in oklch, var(--accent) 40%, transparent)",
  },
  ".cm-scroller": {
    fontFamily: "var(--font-mono)",
    lineHeight: "32px",
    overflowX: "auto",
    overflowY: "hidden",
  },
  ".cm-content": { padding: "0 12px 0 32px", caretColor: "var(--fg)" },
  ".cm-line": { padding: "0" },
  ".cm-placeholder": { color: "var(--fg-subtle)" },
  // Selection in the editor itself (not the popup).
  "&.cm-focused .cm-selectionBackground, ::selection": {
    background: "color-mix(in oklch, var(--accent) 30%, transparent)",
  },
}, { dark: true }); // tell CodeMirror this is a dark theme → dark popups/selection

export function CqlInput({ value, onChange, onRun, disabled, placeholder, className }: CqlInputProps) {
  const host = useRef<HTMLDivElement | null>(null);
  const view = useRef<EditorView | null>(null);
  // Latest callbacks, so the once-built keymap/listener always call current props.
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const onRunRef = useRef(onRun);
  onRunRef.current = onRun;
  const placeholderComp = useRef(new Compartment());
  const editableComp = useRef(new Compartment());

  // Build the editor once.
  useEffect(() => {
    if (!host.current) return;
    const state = EditorState.create({
      doc: value,
      extensions: [
        keymap.of([
          { key: "Enter", run: () => (onRunRef.current(), true), preventDefault: true },
          { key: "Mod-Enter", run: () => (onRunRef.current(), true), preventDefault: true },
          ...closeBracketsKeymap,
          ...completionKeymap,
          ...historyKeymap,
          ...defaultKeymap,
        ]),
        history(),
        closeBrackets(),
        bracketMatching(),
        autocompletion({ override: [cqlCompletion] }),
        cqlHighlight,
        cqlLinter,
        singleLine,
        theme,
        placeholderComp.current.of(cmPlaceholder(placeholder ?? "")),
        editableComp.current.of([
          EditorView.editable.of(!disabled),
          EditorState.readOnly.of(!!disabled),
        ]),
        EditorView.updateListener.of((u) => {
          if (u.docChanged) onChangeRef.current(u.state.doc.toString());
        }),
      ],
    });
    const v = new EditorView({ state, parent: host.current });
    view.current = v;
    return () => {
      v.destroy();
      view.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sync external value changes (e.g. recent-query click) without loops.
  useEffect(() => {
    const v = view.current;
    if (!v) return;
    const cur = v.state.doc.toString();
    if (cur !== value) {
      v.dispatch({ changes: { from: 0, to: cur.length, insert: value } });
    }
  }, [value]);

  useEffect(() => {
    view.current?.dispatch({
      effects: placeholderComp.current.reconfigure(cmPlaceholder(placeholder ?? "")),
    });
  }, [placeholder]);

  useEffect(() => {
    view.current?.dispatch({
      effects: editableComp.current.reconfigure([
        EditorView.editable.of(!disabled),
        EditorState.readOnly.of(!!disabled),
      ]),
    });
  }, [disabled]);

  return <div ref={host} className={className} />;
}
