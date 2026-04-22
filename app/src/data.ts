// Placeholder fixtures — will be replaced with Tauri IPC calls once the
// backend commands in app/src-tauri/src/commands.rs are fleshed out.
// Shape mirrors types.ts exactly.

import type {
  Collocate,
  CorpusMeta,
  DocFreqRow,
  DocumentMeta,
  ExpandedHit,
  FreqRow,
  KwicHit,
  QueryLayer,
  RecentQuery,
} from "./types";

export const CORPORA: CorpusMeta[] = [
  {
    id: "gut-en",
    name: "Gutenberg · EN",
    kind: "literary",
    indexPath: "~/corpora/gut-en/index",
    sourcePath: "~/corpora/gut-en",
    annotated: true,
    docCount: 544,
    tokenCount: 79_467_311,
    types: 412_908,
    avgDocLen: 146_083,
    builtAt: "2026-04-14T10:12:03Z",
    buildMs: 211_000,
    languages: ["en"],
    tokeniser: "corpust-v0.6 · default",
    annotator: "TreeTagger · english-utf8.par",
    sizeOnDisk: 612_000_000,
  },
  {
    id: "scotus",
    name: "US Supreme Court",
    kind: "legal",
    indexPath: "~/corpora/scotus/index",
    sourcePath: "~/corpora/scotus",
    annotated: true,
    docCount: 27_442,
    tokenCount: 214_883_102,
    types: 288_441,
    avgDocLen: 7_831,
    builtAt: "2026-04-02T08:47:11Z",
    buildMs: 842_000,
    languages: ["en"],
    tokeniser: "corpust-v0.6 · legal",
    annotator: "TreeTagger · english-utf8.par",
    sizeOnDisk: 1_720_000_000,
  },
  {
    id: "nyt-2020s",
    name: "NYT · 2020–2025",
    kind: "news",
    indexPath: "~/corpora/nyt-2020s/index",
    sourcePath: "~/corpora/nyt-2020s",
    annotated: false,
    docCount: 418_204,
    tokenCount: 932_481_221,
    types: 1_044_219,
    avgDocLen: 2_231,
    builtAt: "2026-03-22T12:15:40Z",
    buildMs: 198_000,
    languages: ["en"],
    tokeniser: "corpust-v0.6 · default",
    annotator: null,
    sizeOnDisk: 3_200_000_000,
  },
  {
    id: "bnc",
    name: "BNC · sample",
    kind: "mixed",
    indexPath: "~/corpora/bnc/index",
    sourcePath: "~/corpora/bnc",
    annotated: true,
    docCount: 4_054,
    tokenCount: 4_144_829,
    types: 88_201,
    avgDocLen: 1_022,
    builtAt: "2026-02-19T22:05:00Z",
    buildMs: 18_400,
    languages: ["en"],
    tokeniser: "corpust-v0.6 · default",
    annotator: "TreeTagger · english-utf8.par",
    sizeOnDisk: 31_400_000,
  },
];

export const RECENT_QUERIES: RecentQuery[] = [
  { id: 1, layer: "word", term: "linguistic", hits: 1284, corpus: "gut-en" },
  { id: 2, layer: "lemma", term: "go", hits: 48_412, corpus: "gut-en" },
  { id: 3, layer: "pos", term: "JJ", hits: 3_102_884, corpus: "gut-en" },
  { id: 4, layer: "word", term: "reasonable", hits: 9_841, corpus: "scotus" },
];

const KWIC: Record<string, KwicHit[]> = {
  "gut-en|word|linguistic": [
    { docId: "austen-pp.txt", pos: 48219, left: "she had chosen with an eye entirely", hit: "linguistic", right: "and social — a fatal combination in her friend," },
    { docId: "joyce-ul.txt", pos: 9012, left: "the question of whether all such", hit: "linguistic", right: "borrowings remain naturalized remains, he conceded," },
    { docId: "orwell-84.txt", pos: 12110, left: "the whole aim of Newspeak was to narrow the range of", hit: "linguistic", right: "and therefore cognitive manoeuvre available to" },
    { docId: "eliot-mm.txt", pos: 77012, left: "a phrase whose force depends on", hit: "linguistic", right: "precision rather than on sentiment or" },
    { docId: "woolf-mrd.txt", pos: 5821, left: "there was a way of speaking, a kind of", hit: "linguistic", right: "signature, that she recognised at once across" },
    { docId: "conrad-hd.txt", pos: 14802, left: "the man had mastered a strange", hit: "linguistic", right: "economy which we could only envy and" },
    { docId: "melville-mb.txt", pos: 8210, left: "Queequeg, in his own barbarous", hit: "linguistic", right: "fashion, would often mutter a charm under" },
    { docId: "carroll-aw.txt", pos: 412, left: "the Mock Turtle was obsessed with", hit: "linguistic", right: "distinctions — reeling, writhing, fainting in coils" },
    { docId: "dickens-tc.txt", pos: 18234, left: "Mr Stryver affected a bluntness that was more", hit: "linguistic", right: "pose than temperament and he knew it" },
    { docId: "tolkien-ht.txt", pos: 2012, left: "the elves of Rivendell preserved a", hit: "linguistic", right: "inheritance older than any kingdom in Middle" },
    { docId: "hardy-tb.txt", pos: 28102, left: "Tess was conscious, dimly, of the", hit: "linguistic", right: "gulf between herself and the parson’s son" },
    { docId: "stevenson-jh.txt", pos: 4119, left: "Jekyll’s correspondence showed a curious", hit: "linguistic", right: "symmetry between the two hands he wrote" },
  ],
  "gut-en|word|that": [
    { docId: "austen-pp.txt", pos: 12, left: "it is a truth universally acknowledged", hit: "that", right: "a single man in possession of a good fortune" },
    { docId: "melville-mb.txt", pos: 88, left: "call me Ishmael. Some years ago — never mind how long", hit: "that", right: ", having little or no money in my purse" },
    { docId: "carroll-aw.txt", pos: 44, left: "Alice was beginning to get very tired of", hit: "that", right: "sitting by her sister on the bank, and of" },
    { docId: "dickens-tc.txt", pos: 8, left: "it was the best of times, it was", hit: "that", right: "the worst of times, it was the age of" },
    { docId: "joyce-ul.txt", pos: 11, left: "stately, plump Buck Mulligan came from", hit: "that", right: "stairhead bearing a bowl of lather on which" },
  ],
  "gut-en|lemma|go": [
    { docId: "austen-pp.txt", pos: 19411, left: "I do not think Jane will", hit: "go", right: "with us to Netherfield at all, for the", lemma: "go", pos_tag: "VB" },
    { docId: "eliot-mm.txt", pos: 4018, left: "Mr Brooke had already decided to", hit: "going", right: "with them as far as the fence of the park", lemma: "go", pos_tag: "VBG" },
    { docId: "carroll-aw.txt", pos: 881, left: "the White Rabbit had just", hit: "went", right: "through the hedge on the far side of the", lemma: "go", pos_tag: "VBD" },
    { docId: "joyce-ul.txt", pos: 2102, left: "we shall — as men — simply", hit: "go", right: "and meet them at the quay as arranged", lemma: "go", pos_tag: "VB" },
    { docId: "conrad-hd.txt", pos: 15811, left: "Kurtz, I learned much later, had", hit: "gone", right: "upriver quite alone that first season and", lemma: "go", pos_tag: "VBN" },
    { docId: "orwell-84.txt", pos: 9801, left: "Winston felt that if he did not", hit: "go", right: "now, he would never go at all — not tomorrow", lemma: "go", pos_tag: "VB" },
    { docId: "woolf-mrd.txt", pos: 1220, left: "Clarissa, half-remembering, half-", hit: "going", right: "over it again, crossed Victoria Street at", lemma: "go", pos_tag: "VBG" },
    { docId: "melville-mb.txt", pos: 14002, left: "Starbuck refused to", hit: "go", right: "aloft while the gale held its present force", lemma: "go", pos_tag: "VB" },
  ],
  "gut-en|pos|NN": [
    { docId: "austen-pp.txt", pos: 35, left: "a single man in possession of a good", hit: "fortune", right: "must be in want of a wife", lemma: "fortune", pos_tag: "NN" },
    { docId: "dickens-tc.txt", pos: 42, left: "it was the age of", hit: "wisdom", right: ", it was the age of foolishness", lemma: "wisdom", pos_tag: "NN" },
    { docId: "melville-mb.txt", pos: 201, left: "it is a way I have of driving off the", hit: "spleen", right: "and regulating the circulation", lemma: "spleen", pos_tag: "NN" },
    { docId: "joyce-ul.txt", pos: 310, left: "he held the bowl aloft and intoned a", hit: "parody", right: "of the introit as he descended", lemma: "parody", pos_tag: "NN" },
    { docId: "woolf-mrd.txt", pos: 1800, left: "the leaden circles dissolved in the", hit: "air", right: ". It was June. The War was over —", lemma: "air", pos_tag: "NN" },
    { docId: "carroll-aw.txt", pos: 90, left: "the rabbit had pulled a gold", hit: "watch", right: "out of its waistcoat pocket and", lemma: "watch", pos_tag: "NN" },
    { docId: "conrad-hd.txt", pos: 2011, left: "the silence of the", hit: "forest", right: "had something terrifying in it,", lemma: "forest", pos_tag: "NN" },
    { docId: "tolkien-ht.txt", pos: 512, left: "Bilbo suddenly remembered the", hit: "ring", right: "in his pocket and felt cold all", lemma: "ring", pos_tag: "NN" },
  ],
};

export const EXPANDED: Record<string, ExpandedHit> = {
  "gut-en|austen-pp.txt|48219": {
    before:
      "Lady Catherine was one of those matrons whose excellence could only be fully appreciated by those whom she had relieved of some inconvenience — which is to say, almost no one. Mrs. Bennet nodded vigorously whenever the name was spoken, and declared that ",
    match: "she had chosen with an eye entirely linguistic and social",
    after:
      " — a fatal combination in her friend, who now watched the proceedings from the window with something between amusement and despair. Elizabeth, arriving late and without ceremony, found the room already arranged into those small parties which her mother took for the natural order of any evening.",
    docTitle: "Pride and Prejudice · Jane Austen",
    docMeta: "1813 · ch. XXXVI · sentence 1,419",
  },
};

export const COLLOCATIONS: Collocate[] = [
  { word: "signature", pos: "NN", leftCount: 2, rightCount: 44, total: 46, logDice: 11.2, mi: 8.4, z: 18.4, dist: 1 },
  { word: "precision", pos: "NN", leftCount: 1, rightCount: 38, total: 39, logDice: 10.8, mi: 7.9, z: 16.9, dist: 1 },
  { word: "economy", pos: "NN", leftCount: 0, rightCount: 29, total: 29, logDice: 10.3, mi: 7.2, z: 14.1, dist: 1 },
  { word: "inheritance", pos: "NN", leftCount: 0, rightCount: 22, total: 22, logDice: 9.9, mi: 6.8, z: 12.0, dist: 1 },
  { word: "turn", pos: "NN", leftCount: 18, rightCount: 2, total: 20, logDice: 9.1, mi: 5.4, z: 9.4, dist: -1 },
  { word: "peculiar", pos: "JJ", leftCount: 14, rightCount: 1, total: 15, logDice: 8.6, mi: 5.1, z: 8.2, dist: -1 },
  { word: "purely", pos: "RB", leftCount: 13, rightCount: 0, total: 13, logDice: 8.4, mi: 5.0, z: 7.8, dist: -1 },
  { word: "fashion", pos: "NN", leftCount: 0, rightCount: 12, total: 12, logDice: 8.1, mi: 4.8, z: 7.1, dist: 1 },
  { word: "distinctions", pos: "NNS", leftCount: 0, rightCount: 11, total: 11, logDice: 9.5, mi: 6.2, z: 10.2, dist: 1 },
  { word: "borrowings", pos: "NNS", leftCount: 0, rightCount: 8, total: 8, logDice: 10.1, mi: 7.4, z: 13.2, dist: 1 },
  { word: "range", pos: "NN", leftCount: 7, rightCount: 0, total: 7, logDice: 7.6, mi: 4.1, z: 6.0, dist: -1 },
];

export const DOC_FREQ: DocFreqRow[] = [
  { doc: "joyce-ul.txt", hits: 312, per1m: 81.2 },
  { doc: "woolf-mrd.txt", hits: 201, per1m: 74.1 },
  { doc: "eliot-mm.txt", hits: 288, per1m: 42.5 },
  { doc: "orwell-84.txt", hits: 102, per1m: 38.2 },
  { doc: "conrad-hd.txt", hits: 58, per1m: 32.0 },
  { doc: "austen-pp.txt", hits: 112, per1m: 28.7 },
  { doc: "melville-mb.txt", hits: 88, per1m: 21.4 },
  { doc: "dickens-tc.txt", hits: 62, per1m: 18.3 },
];

export const DOCUMENTS: DocumentMeta[] = [
  { id: "austen-pp.txt", title: "Pride and Prejudice", author: "Jane Austen", year: 1813, tokens: 122_189 },
  { id: "austen-ss.txt", title: "Sense and Sensibility", author: "Jane Austen", year: 1811, tokens: 119_302 },
  { id: "carroll-aw.txt", title: "Alice in Wonderland", author: "Lewis Carroll", year: 1865, tokens: 26_509 },
  { id: "conrad-hd.txt", title: "Heart of Darkness", author: "Joseph Conrad", year: 1899, tokens: 38_112 },
  { id: "dickens-tc.txt", title: "A Tale of Two Cities", author: "Charles Dickens", year: 1859, tokens: 136_058 },
  { id: "eliot-mm.txt", title: "Middlemarch", author: "George Eliot", year: 1871, tokens: 316_059 },
  { id: "hardy-tb.txt", title: "Tess of the d’Urbervilles", author: "Thomas Hardy", year: 1891, tokens: 151_830 },
  { id: "joyce-ul.txt", title: "Ulysses", author: "James Joyce", year: 1922, tokens: 264_911 },
  { id: "melville-mb.txt", title: "Moby-Dick", author: "Herman Melville", year: 1851, tokens: 209_117 },
  { id: "orwell-84.txt", title: "Nineteen Eighty-Four", author: "George Orwell", year: 1949, tokens: 103_499 },
  { id: "stevenson-jh.txt", title: "Dr. Jekyll and Mr. Hyde", author: "R.L. Stevenson", year: 1886, tokens: 26_102 },
  { id: "tolkien-ht.txt", title: "The Hobbit", author: "J.R.R. Tolkien", year: 1937, tokens: 95_022 },
  { id: "woolf-mrd.txt", title: "Mrs Dalloway", author: "Virginia Woolf", year: 1925, tokens: 63_410 },
];

export const POS_FREQ: FreqRow[] = [
  { tag: "NN", label: "singular noun", count: 12_408_221, pct: 15.6 },
  { tag: "IN", label: "preposition", count: 9_901_408, pct: 12.5 },
  { tag: "DT", label: "determiner", count: 8_214_003, pct: 10.3 },
  { tag: "JJ", label: "adjective", count: 5_891_302, pct: 7.4 },
  { tag: "VBD", label: "verb, past", count: 4_120_884, pct: 5.2 },
  { tag: "NNS", label: "plural noun", count: 3_902_118, pct: 4.9 },
  { tag: "PRP", label: "pronoun, personal", count: 3_489_112, pct: 4.4 },
  { tag: "VB", label: "verb, base", count: 2_912_083, pct: 3.7 },
  { tag: "RB", label: "adverb", count: 2_104_508, pct: 2.6 },
  { tag: "CC", label: "coordinator", count: 1_982_411, pct: 2.5 },
];

export const WORD_FREQ: FreqRow[] = [
  { word: "the", count: 5_214_003, pct: 6.56 },
  { word: "of", count: 2_892_881, pct: 3.64 },
  { word: "and", count: 2_618_122, pct: 3.29 },
  { word: "to", count: 2_401_998, pct: 3.02 },
  { word: "a", count: 1_938_214, pct: 2.44 },
  { word: "in", count: 1_712_022, pct: 2.15 },
  { word: "he", count: 1_100_881, pct: 1.38 },
  { word: "was", count: 982_401, pct: 1.24 },
  { word: "that", count: 912_118, pct: 1.15 },
  { word: "his", count: 844_502, pct: 1.06 },
];

export function pickHits(corpusId: string, term: string, layer: QueryLayer): KwicHit[] {
  const key = `${corpusId}|${layer}|${term.toLowerCase()}`;
  if (KWIC[key]) return KWIC[key];
  const prefix = `${corpusId}|${layer}|`;
  const fallback = Object.keys(KWIC).find((k) => k.startsWith(prefix));
  const base = fallback ? KWIC[fallback] : KWIC["gut-en|word|that"];
  return base.map((h) => ({ ...h, hit: term }));
}

export function makeDispersion(seed: number): number[] {
  const n = 100;
  const out: number[] = [];
  let s = seed;
  for (let i = 0; i < 42; i++) {
    s = (s * 9301 + 49297) % 233280;
    out.push(Math.floor((s / 233280) * n));
  }
  return out;
}
