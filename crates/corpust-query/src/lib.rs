//! Query layer.
//!
//! Phase 0 is a thin façade over [`CorpusIndex::kwic`] — the point of the
//! crate is to establish the seam. The CQL parser lands here in a later phase;
//! at that point the façade grows a real query-plan pipeline without changing
//! callers' import paths.

use anyhow::{Result, anyhow, bail};
use corpust_index::{
    AttrConstraint, CorpusIndex, CqlQuery, DEFAULT_CONTEXT, DEFAULT_LIMIT, DocFilter, KwicPage,
    Matcher, QueryLayer, TokenPattern,
};
use regex::RegexBuilder;

pub use corpust_index::{
    AttrConstraint as CqlConstraint, CqlQuery as Query, DocFilter as Filter, Matcher as CqlMatcher,
    QueryLayer as Layer, TokenPattern as CqlToken,
};

/// Parameters for a KWIC query.
#[derive(Debug, Clone)]
pub struct KwicRequest<'a> {
    pub term: &'a str,
    pub layer: QueryLayer,
    pub context: usize,
    pub limit: usize,
    /// Hits to skip before the page — for concordance pagination.
    pub offset: usize,
    /// Document-metadata filter; an empty filter (the default) matches
    /// the whole corpus.
    pub filter: DocFilter,
}

impl<'a> KwicRequest<'a> {
    pub fn new(term: &'a str) -> Self {
        Self {
            term,
            layer: QueryLayer::Word,
            context: DEFAULT_CONTEXT,
            limit: DEFAULT_LIMIT,
            offset: 0,
            filter: DocFilter::default(),
        }
    }

    pub fn layer(mut self, layer: QueryLayer) -> Self {
        self.layer = layer;
        self
    }

    pub fn filter(mut self, filter: DocFilter) -> Self {
        self.filter = filter;
        self
    }

    pub fn context(mut self, context: usize) -> Self {
        self.context = context;
        self
    }

    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }
}

pub fn kwic(index: &CorpusIndex, request: KwicRequest<'_>) -> Result<KwicPage> {
    let filter = (!request.filter.is_empty()).then_some(&request.filter);
    if is_cql(request.term) {
        let query = parse_cql(request.term)?;
        index.cql_kwic(
            &query,
            request.context,
            request.limit,
            request.offset,
            filter,
        )
    } else {
        index.kwic_filtered(
            request.term,
            request.layer,
            request.context,
            request.limit,
            request.offset,
            filter,
        )
    }
}

// ---------------------------------------------------------------------------
// CQL — corpus query language
// ---------------------------------------------------------------------------
//
// Grammar (phase 1):
//   query  := token+
//   token  := '[' constraint (ws constraint)* ']' | '"' value '"'
//   constraint := attr '=' '"' value '"'
//   attr   := word | lemma | pos   (aliases: hw→lemma, tag→pos)
// A bare quoted token `"x"` is sugar for `[word="x"]`. Values are exact
// when they contain only word characters, otherwise compiled as a
// full-token-anchored regex (case-insensitive for word/lemma).

/// True when `input` should be parsed as CQL rather than a bare term: it
/// contains a `[` token, or starts with a quote. Bare terms keep the
/// classic single-layer behaviour.
pub fn is_cql(input: &str) -> bool {
    let t = input.trim_start();
    t.starts_with('[') || t.starts_with('"') || input.contains('[')
}

/// Parse a CQL string into a [`CqlQuery`]. Returns a descriptive error on
/// malformed input (the UI surfaces it in place of results).
pub fn parse_cql(input: &str) -> Result<CqlQuery> {
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    let mut tokens = Vec::new();
    skip_ws(&chars, &mut i);
    while i < chars.len() {
        match chars[i] {
            '[' => tokens.push(parse_bracket_token(&chars, &mut i)?),
            '"' => {
                let value = parse_quoted(&chars, &mut i)?;
                tokens.push(TokenPattern {
                    constraints: vec![make_constraint(QueryLayer::Word, &value)?],
                });
            }
            c => bail!("unexpected `{c}` — expected a `[…]` token or a quoted string"),
        }
        skip_ws(&chars, &mut i);
    }
    if tokens.is_empty() {
        bail!("empty query");
    }
    Ok(CqlQuery { tokens })
}

fn skip_ws(chars: &[char], i: &mut usize) {
    while *i < chars.len() && chars[*i].is_whitespace() {
        *i += 1;
    }
}

fn parse_bracket_token(chars: &[char], i: &mut usize) -> Result<TokenPattern> {
    *i += 1; // consume '['
    let mut constraints = Vec::new();
    loop {
        skip_ws(chars, i);
        match chars.get(*i) {
            None => bail!("unterminated `[` — missing `]`"),
            Some(']') => {
                *i += 1;
                break;
            }
            Some(c) if c.is_alphabetic() => {
                let attr = parse_ident(chars, i);
                let layer = resolve_attr(&attr)?;
                skip_ws(chars, i);
                if chars.get(*i) != Some(&'=') {
                    bail!("expected `=` after attribute `{attr}`");
                }
                *i += 1; // consume '='
                skip_ws(chars, i);
                if chars.get(*i) != Some(&'"') {
                    bail!("expected a quoted value after `{attr}=`");
                }
                let value = parse_quoted(chars, i)?;
                constraints.push(make_constraint(layer, &value)?);
            }
            Some(c) => bail!("unexpected `{c}` inside `[…]` — expected an attribute name or `]`"),
        }
    }
    if constraints.is_empty() {
        bail!("empty token `[]` — give it at least one attribute");
    }
    Ok(TokenPattern { constraints })
}

fn parse_ident(chars: &[char], i: &mut usize) -> String {
    let start = *i;
    while *i < chars.len() && (chars[*i].is_alphanumeric() || chars[*i] == '_') {
        *i += 1;
    }
    chars[start..*i].iter().collect()
}

fn parse_quoted(chars: &[char], i: &mut usize) -> Result<String> {
    *i += 1; // consume opening '"'
    let start = *i;
    while *i < chars.len() && chars[*i] != '"' {
        *i += 1;
    }
    if chars.get(*i) != Some(&'"') {
        bail!("unterminated quoted value");
    }
    let value: String = chars[start..*i].iter().collect();
    *i += 1; // consume closing '"'
    Ok(value)
}

fn resolve_attr(attr: &str) -> Result<QueryLayer> {
    match attr {
        "word" => Ok(QueryLayer::Word),
        "lemma" | "hw" => Ok(QueryLayer::Lemma),
        "pos" | "tag" => Ok(QueryLayer::Pos),
        other => Err(anyhow!(
            "unknown attribute `{other}` — use word, lemma (hw), or pos (tag)"
        )),
    }
}

/// Build a constraint, choosing exact vs regex matching. A value that is
/// only word characters matches exactly (fast, single dict lookup);
/// anything else compiles to a full-token-anchored regex. Word/lemma are
/// case-insensitive (the index lowercases them); pos is case-sensitive.
fn make_constraint(layer: QueryLayer, value: &str) -> Result<AttrConstraint> {
    if value.is_empty() {
        bail!("empty value");
    }
    let case_insensitive = !matches!(layer, QueryLayer::Pos);
    let is_plain = value
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-');
    let matcher = if is_plain {
        let text = if case_insensitive {
            value.to_lowercase()
        } else {
            value.to_string()
        };
        Matcher::Exact(text)
    } else {
        let re = RegexBuilder::new(&format!("^(?:{value})$"))
            .case_insensitive(case_insensitive)
            .size_limit(1 << 20)
            .build()
            .map_err(|e| anyhow!("invalid regex `{value}`: {e}"))?;
        Matcher::Regex(re)
    };
    Ok(AttrConstraint { layer, matcher })
}

#[cfg(test)]
mod tests {
    use super::*;
    use corpust_core::Document;
    use std::path::PathBuf;

    fn tiny_index() -> (tempfile::TempDir, CorpusIndex) {
        let tmp = tempfile::tempdir().unwrap();
        let idx = CorpusIndex::create(tmp.path()).unwrap();
        idx.add_documents(
            [Document {
                id: 0,
                path: PathBuf::from("a.txt"),
                text: "the quick brown fox jumps over the lazy dog".to_string(),
            }],
            None,
        )
        .unwrap();
        (tmp, idx)
    }

    #[test]
    fn builder_defaults() {
        let req = KwicRequest::new("foo");
        assert_eq!(req.term, "foo");
        assert!(matches!(req.layer, QueryLayer::Word));
        assert_eq!(req.context, DEFAULT_CONTEXT);
        assert_eq!(req.limit, DEFAULT_LIMIT);
    }

    #[test]
    fn builder_overrides() {
        let req = KwicRequest::new("foo")
            .layer(QueryLayer::Lemma)
            .context(7)
            .limit(3);
        assert!(matches!(req.layer, QueryLayer::Lemma));
        assert_eq!(req.context, 7);
        assert_eq!(req.limit, 3);
    }

    #[test]
    fn kwic_facade_returns_index_hits() {
        let (_tmp, idx) = tiny_index();
        let page = kwic(&idx, KwicRequest::new("the").context(2).limit(10)).unwrap();
        assert_eq!(page.total, 2);
        assert_eq!(page.hits.len(), 2);
        assert!(page.hits.iter().all(|h| h.hit == "the"));
    }

    #[test]
    fn kwic_facade_paginates() {
        let (_tmp, idx) = tiny_index();
        // "the" occurs twice; page size 1 yields one hit per page, total 2.
        let p0 = kwic(&idx, KwicRequest::new("the").context(2).limit(1).offset(0)).unwrap();
        assert_eq!(p0.total, 2);
        assert_eq!(p0.hits.len(), 1);
        let p1 = kwic(&idx, KwicRequest::new("the").context(2).limit(1).offset(1)).unwrap();
        assert_eq!(p1.total, 2);
        assert_eq!(p1.hits.len(), 1);
        assert_ne!(p0.hits[0].hit_position, p1.hits[0].hit_position);
    }

    #[test]
    fn is_cql_detects_bracket_and_quote_forms() {
        assert!(is_cql("[pos=\"NN\"]"));
        assert!(is_cql("\"bank\""));
        assert!(is_cql("[word=\"a\"] [word=\"b\"]"));
        assert!(!is_cql("bank"));
        assert!(!is_cql("run.*")); // bare regex term stays a classic query
    }

    #[test]
    fn parse_single_token_multi_attribute() {
        let q = parse_cql("[word=\"bank\" pos=\"NN\"]").unwrap();
        assert_eq!(q.tokens.len(), 1);
        assert_eq!(q.tokens[0].constraints.len(), 2);
        assert!(matches!(q.tokens[0].constraints[0].layer, QueryLayer::Word));
        assert!(matches!(q.tokens[0].constraints[1].layer, QueryLayer::Pos));
        // "bank" is plain → exact (and lowercased); "NN" is plain → exact.
        assert!(matches!(&q.tokens[0].constraints[0].matcher, Matcher::Exact(t) if t == "bank"));
    }

    #[test]
    fn parse_sequence_aliases_and_bare_quote() {
        let q = parse_cql("\"the\" [tag=\"NN.*\"] [hw=\"be\"]").unwrap();
        assert_eq!(q.tokens.len(), 3);
        // bare quote → word
        assert!(matches!(q.tokens[0].constraints[0].layer, QueryLayer::Word));
        // tag → pos, value has a metachar → regex
        assert!(matches!(q.tokens[1].constraints[0].layer, QueryLayer::Pos));
        assert!(matches!(
            q.tokens[1].constraints[0].matcher,
            Matcher::Regex(_)
        ));
        // hw → lemma
        assert!(matches!(
            q.tokens[2].constraints[0].layer,
            QueryLayer::Lemma
        ));
    }

    #[test]
    fn parse_errors_are_descriptive() {
        assert!(
            parse_cql("[word=\"a\"")
                .unwrap_err()
                .to_string()
                .contains("]")
        );
        assert!(
            parse_cql("[]")
                .unwrap_err()
                .to_string()
                .contains("empty token")
        );
        assert!(
            parse_cql("[foo=\"x\"]")
                .unwrap_err()
                .to_string()
                .contains("unknown attribute")
        );
        assert!(parse_cql("[word=\"(\"]").is_err()); // invalid regex
    }

    #[test]
    fn cql_kwic_equivalent_to_bare_term_on_word_layer() {
        let (_tmp, idx) = tiny_index();
        let bare = kwic(&idx, KwicRequest::new("the").context(2).limit(10)).unwrap();
        let cql = kwic(
            &idx,
            KwicRequest::new("[word=\"the\"]").context(2).limit(10),
        )
        .unwrap();
        assert_eq!(bare.total, cql.total);
        assert_eq!(bare.hits.len(), cql.hits.len());
        assert!(cql.hits.iter().all(|h| h.hit == "the"));
    }

    #[test]
    fn cql_two_token_sequence_matches_adjacent_positions() {
        let (_tmp, idx) = tiny_index();
        // tiny_index body: "the quick brown fox jumps over the lazy dog"
        let q = kwic(
            &idx,
            KwicRequest::new("[word=\"the\"] [word=\"lazy\"]")
                .context(2)
                .limit(10),
        )
        .unwrap();
        assert_eq!(q.total, 1); // only "the lazy" (not "the quick")
        assert_eq!(q.hits[0].hit, "the lazy");
    }
}
