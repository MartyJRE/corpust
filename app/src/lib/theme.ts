// Theme switching. A theme is just a set of CSS-variable values applied via
// `data-theme` on <html>; every surface (incl. the CQL editor) reads those
// variables, so switching is instant and global. Palettes live in index.css.
//
// `dark` drives CodeMirror's base theme flag (popup/selection chrome), which
// the editor reconfigures on the `corpust-theme` event below.

export interface Theme {
  id: string;
  name: string;
  dark: boolean;
}

/** Built-in themes — popular code-editor palettes. Each `id` has a matching
 *  `:root[data-theme="<id>"]` block in index.css. */
export const THEMES: Theme[] = [
  { id: "corpust", name: "Corpust Dark", dark: true },
  { id: "onedark", name: "One Dark", dark: true },
  { id: "monokai", name: "Monokai", dark: true },
  { id: "dracula", name: "Dracula", dark: true },
  { id: "nord", name: "Nord", dark: true },
  { id: "gruvbox", name: "Gruvbox Dark", dark: true },
  { id: "tokyonight", name: "Tokyo Night", dark: true },
  { id: "catppuccin", name: "Catppuccin Mocha", dark: true },
  { id: "solarized", name: "Solarized Dark", dark: true },
  { id: "onelight", name: "One Light", dark: false },
  { id: "github", name: "GitHub Light", dark: false },
  { id: "solarized-light", name: "Solarized Light", dark: false },
  { id: "latte", name: "Catppuccin Latte", dark: false },
];

const STORAGE_KEY = "corpust-theme";
const DEFAULT_THEME = "corpust";

/** Event fired (on window) when the theme changes, so non-React surfaces
 *  like the CodeMirror editor can react. */
export const THEME_EVENT = "corpust-theme";

/** The persisted theme id, or the default when unset/unknown. */
export function loadTheme(): string {
  try {
    const id = localStorage.getItem(STORAGE_KEY);
    if (id && THEMES.some((t) => t.id === id)) return id;
  } catch {
    // localStorage unavailable (private mode / non-browser) — use default.
  }
  return DEFAULT_THEME;
}

/** Apply a theme: set `data-theme` on <html>, persist, and notify. */
export function applyTheme(id: string): void {
  document.documentElement.dataset.theme = id;
  try {
    localStorage.setItem(STORAGE_KEY, id);
  } catch {
    // ignore persistence failures
  }
  window.dispatchEvent(new CustomEvent(THEME_EVENT, { detail: id }));
}

/** The currently-applied theme id (from the DOM, falling back to storage). */
export function currentThemeId(): string {
  return document.documentElement.dataset.theme || loadTheme();
}

/** Whether a theme is dark (defaults to dark for unknown ids). */
export function isThemeDark(id: string): boolean {
  return THEMES.find((t) => t.id === id)?.dark ?? true;
}
