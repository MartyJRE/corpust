import { beforeEach, describe, expect, it } from "vitest";
import { THEME_EVENT, applyTheme, currentThemeId, isThemeDark, loadTheme } from "./theme";

describe("theme", () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute("data-theme");
  });

  it("defaults to the original Corpust Dark", () => {
    expect(loadTheme()).toBe("corpust");
  });

  it("applyTheme sets the attribute, persists, and fires the event", () => {
    let fired: string | null = null;
    window.addEventListener(THEME_EVENT, (e) => {
      fired = (e as CustomEvent).detail;
    }, { once: true });
    applyTheme("onedark");
    expect(document.documentElement.dataset.theme).toBe("onedark");
    expect(loadTheme()).toBe("onedark");
    expect(currentThemeId()).toBe("onedark");
    expect(fired).toBe("onedark");
  });

  it("ignores an unknown persisted theme", () => {
    localStorage.setItem("corpust-theme", "bogus");
    expect(loadTheme()).toBe("corpust");
  });

  it("isThemeDark reflects metadata (defaulting to dark)", () => {
    expect(isThemeDark("corpust")).toBe(true);
    expect(isThemeDark("github")).toBe(false);
    expect(isThemeDark("latte")).toBe(false);
    expect(isThemeDark("unknown")).toBe(true);
  });
});
