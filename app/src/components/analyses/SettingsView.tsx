import { useState } from "react";

export function SettingsView() {
  const [ann, setAnn] = useState(true);
  const [annotator, setAnnotator] = useState("treetagger");
  const [theme, setTheme] = useState("dark");
  const [ligs, setLigs] = useState(false);

  return (
    <div className="cx-settings">
      <h1>settings</h1>
      <p className="sub">preferences · persisted to ~/.config/corpust/config.toml</p>

      <div className="cx-setting-group">
        <div className="cx-setting-label">
          theme
          <span className="desc">dark is tuned for long reading sessions.</span>
        </div>
        <div className="cx-setting-control">
          <select className="cx-select" value={theme} onChange={(e) => setTheme(e.target.value)}>
            <option value="dark">dark</option>
            <option value="light">light</option>
            <option value="system">follow system</option>
          </select>
        </div>
      </div>

      <div className="cx-setting-group">
        <div className="cx-setting-label">
          annotation on build
          <span className="desc">
            run TreeTagger during index build. ≈18× slower; unlocks lemma + POS queries.
          </span>
        </div>
        <div className="cx-setting-control">
          <div
            className={`cx-toggle ${ann ? "is-on" : ""}`}
            role="switch"
            aria-checked={ann}
            tabIndex={0}
            onClick={() => setAnn(!ann)}
            onKeyDown={(e) => {
              if (e.key === " " || e.key === "Enter") setAnn(!ann);
            }}
          />
        </div>
      </div>

      <div className="cx-setting-group">
        <div className="cx-setting-label">
          annotator
          <span className="desc">external tool for POS + lemma. corpust shells out to it.</span>
        </div>
        <div className="cx-setting-control">
          <select
            className="cx-select"
            value={annotator}
            onChange={(e) => setAnnotator(e.target.value)}
          >
            <option value="treetagger">TreeTagger · english-utf8.par</option>
            <option value="spacy">spaCy · en_core_web_sm</option>
            <option value="stanza">Stanza · en</option>
          </select>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: "var(--fg-subtle)",
              marginTop: 4,
            }}
          >
            resolved: /usr/local/bin/tree-tagger
          </div>
        </div>
      </div>

      <div className="cx-setting-group">
        <div className="cx-setting-label">
          ligatures in KWIC
          <span className="desc">off by default — hit tokens render as typed. recommended: off.</span>
        </div>
        <div className="cx-setting-control">
          <div
            className={`cx-toggle ${ligs ? "is-on" : ""}`}
            role="switch"
            aria-checked={ligs}
            tabIndex={0}
            onClick={() => setLigs(!ligs)}
            onKeyDown={(e) => {
              if (e.key === " " || e.key === "Enter") setLigs(!ligs);
            }}
          />
        </div>
      </div>

      <div className="cx-setting-group">
        <div className="cx-setting-label">
          keyboard shortcuts
          <span className="desc">all shortcuts are customisable in config.toml.</span>
        </div>
        <div className="cx-setting-control">
          <ShortcutRow k="command palette" v={<span className="cx-kbd">⌘K</span>} />
          <ShortcutRow k="build index" v={<span className="cx-kbd">⌘B</span>} />
          <ShortcutRow k="open corpus" v={<span className="cx-kbd">⌘O</span>} />
          <ShortcutRow
            k="switch layer"
            v={
              <>
                <span className="cx-kbd">⌘1</span> / <span className="cx-kbd">⌘2</span> /{" "}
                <span className="cx-kbd">⌘3</span>
              </>
            }
          />
          <ShortcutRow
            k="next hit / prev hit"
            v={
              <>
                <span className="cx-kbd">j</span> / <span className="cx-kbd">k</span>
              </>
            }
          />
          <ShortcutRow k="close drawer" v={<span className="cx-kbd">esc</span>} />
        </div>
      </div>
    </div>
  );
}

function ShortcutRow({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        padding: "2px 0",
        gap: 10,
        fontFamily: "var(--font-mono)",
        fontSize: 11,
      }}
    >
      <span style={{ color: "var(--fg)" }}>{k}</span>
      <span style={{ color: "var(--fg-muted)" }}>{v}</span>
    </div>
  );
}
