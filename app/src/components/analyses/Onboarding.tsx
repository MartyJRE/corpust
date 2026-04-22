import { Folder, Hammer } from "lucide-react";
import { Wordmark } from "@/components/chrome/Wordmark";

export interface OnboardingProps {
  onOpen: () => void;
  onBuild: () => void;
  onSample: () => void;
}

export function Onboarding({ onOpen, onBuild, onSample }: OnboardingProps) {
  return (
    <div className="cx-onboard">
      <div className="cx-onboard-card">
        <Wordmark size="lg" />
        <div className="cx-onboard-lede">
          A fast corpus-linguistics toolkit for researchers. Index billions of
          tokens locally; query with concordance, lemma, POS and collocation
          views.
        </div>
        <div className="cx-onboard-options">
          <button type="button" className="cx-onboard-opt" onClick={onBuild}>
            <div className="h">
              <Hammer size={14} />
              Build an index
            </div>
            <div className="d">
              Point corpust at a folder of <span style={{ fontFamily: "var(--font-mono)" }}>.txt</span> files. Indexes ~4.5M words/sec.
            </div>
            <div className="cmd">$ corpust build ./corpora/my-corpus</div>
          </button>
          <button type="button" className="cx-onboard-opt" onClick={onOpen}>
            <div className="h">
              <Folder size={14} />
              Open existing
            </div>
            <div className="d">Load a previously-built index.</div>
            <div className="cmd">$ corpust open ./my-corpus/index</div>
          </button>
        </div>
        <div className="cx-onboard-sample">
          <span style={{ color: "var(--fg-subtle)" }}>no data yet?</span>
          <a onClick={onSample}>try the Gutenberg sample (79M words)</a>
        </div>
      </div>
    </div>
  );
}
