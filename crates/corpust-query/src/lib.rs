//! Query layer.
//!
//! Phase 0 is a thin façade over [`CorpusIndex::kwic`] — the point of the
//! crate is to establish the seam. The CQL parser lands here in a later phase;
//! at that point the façade grows a real query-plan pipeline without changing
//! callers' import paths.

use anyhow::Result;
use corpust_index::{CorpusIndex, DEFAULT_CONTEXT, DEFAULT_LIMIT, KwicHit, QueryLayer};

pub use corpust_index::QueryLayer as Layer;

/// Parameters for a KWIC query.
#[derive(Debug, Clone)]
pub struct KwicRequest<'a> {
    pub term: &'a str,
    pub layer: QueryLayer,
    pub context: usize,
    pub limit: usize,
}

impl<'a> KwicRequest<'a> {
    pub fn new(term: &'a str) -> Self {
        Self {
            term,
            layer: QueryLayer::Word,
            context: DEFAULT_CONTEXT,
            limit: DEFAULT_LIMIT,
        }
    }

    pub fn layer(mut self, layer: QueryLayer) -> Self {
        self.layer = layer;
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
}

pub fn kwic(index: &CorpusIndex, request: KwicRequest<'_>) -> Result<Vec<KwicHit>> {
    index.kwic(request.term, request.layer, request.context, request.limit)
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
        let hits = kwic(&idx, KwicRequest::new("the").context(2).limit(10)).unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|h| h.hit == "the"));
    }
}
