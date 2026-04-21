//! Query layer.
//!
//! Phase 0 is a thin façade over [`CorpusIndex::kwic`] — the point of the
//! crate is to establish the seam. The CQL parser lands here in a later phase;
//! at that point the façade grows a real query-plan pipeline without changing
//! callers' import paths.

use anyhow::Result;
use corpust_index::{CorpusIndex, DEFAULT_CONTEXT, DEFAULT_LIMIT, KwicHit};

/// Parameters for a KWIC query.
#[derive(Debug, Clone)]
pub struct KwicRequest<'a> {
    pub term: &'a str,
    pub context: usize,
    pub limit: usize,
}

impl<'a> KwicRequest<'a> {
    pub fn new(term: &'a str) -> Self {
        Self {
            term,
            context: DEFAULT_CONTEXT,
            limit: DEFAULT_LIMIT,
        }
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
    index.kwic(request.term, request.context, request.limit)
}
