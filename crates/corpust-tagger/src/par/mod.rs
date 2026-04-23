//! `.par` file reader.
//!
//! The format is undocumented. Schmid's docs note only that parameter
//! files are platform- and version-specific, which is exactly what you'd
//! expect from naive `fwrite(&struct, ...)` serialization. A hex dump of
//! the bundled English model confirms:
//!
//! - Native-endian (little-endian on our reference file) `u32` counts.
//! - Null-terminated C strings for tag names and (later) lexicon words.
//! - No magic bytes and no internal version field; version lives in the
//!   filename (`3.2`).
//!
//! We learn the layout one section at a time. Each module under this
//! one owns a single section and a differential test that diffs its
//! output against what the reference binary produces for the same input.

pub mod dtree;
pub mod header;
pub mod lexicon;

use anyhow::{Context, Result};
use std::path::Path;

/// Fully loaded parameter model.
///
/// Grows field-by-field as each section is implemented. Fields that
/// aren't reverse-engineered yet simply aren't here — we don't pretend
/// to have read data we haven't read.
#[derive(Debug, Clone)]
pub struct Model {
    pub header: header::Header,
    pub lexicon: lexicon::Lexicon,
    /// Decision-tree section — records in file order, kind-tagged.
    /// Populated when the file's tail parses cleanly as a dtree
    /// section; a load that succeeds without this field means the
    /// walker found unrecognised bytes and we've stored the error so
    /// callers can keep using the header/lexicon while archaeology
    /// continues. Intentional looseness: the tries between the lexicon
    /// and the dtree are still undecoded, so we *can't* seek straight
    /// to the dtree from the lexicon's end cursor yet.
    pub dtree: Option<dtree::DecisionTree>,
}

/// Entry point: memory-map (eventually) and parse a `.par` file.
///
/// For now we read the whole file into memory. The English model is
/// ~14 MB and only loaded once per process, so this is fine; if we
/// ever want to `mmap` it we can swap this without touching callers.
pub fn load(path: &Path) -> Result<Model> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading .par file {}", path.display()))?;
    let mut cur = Cursor::new(&bytes);
    let header = header::read(&mut cur).context("parsing .par header")?;
    let lexicon = lexicon::read(&mut cur, &header).context("parsing .par lexicon")?;
    // Dtree loader is stand-alone: it takes a cursor already
    // positioned at the dtree start. For `english.par` that's
    // `0xd231bb`, a known constant until the tries get a proper
    // reader. Other `.par` files would need their own positioning.
    // Gated on `english.par`'s known offset so other files don't
    // silently pull garbage through the dtree walker.
    let dtree = try_read_english_dtree(&bytes, &header);
    Ok(Model { header, lexicon, dtree })
}

/// Best-effort dtree load for the bundled `english.par`. Returns
/// `None` if the file isn't the exact shape we've reverse-engineered
/// so far (58 tags, dtree at `0xd231bb`), or if the walker rejects
/// some bytes — the header + lexicon are still useful in that case,
/// so we don't propagate the error.
fn try_read_english_dtree(
    bytes: &[u8],
    header: &header::Header,
) -> Option<dtree::DecisionTree> {
    const ENGLISH_DTREE_START: usize = 0xd231bb;
    if header.tags.len() != 58 || bytes.len() <= ENGLISH_DTREE_START {
        return None;
    }
    let mut cur = Cursor::new(bytes);
    cur.advance(ENGLISH_DTREE_START).ok()?;
    dtree::read(&mut cur, header).ok()
}

/// Minimal byte-level cursor for `.par` parsing.
///
/// Intentionally tiny. `byteorder` + `std::io::Cursor` would be fine too,
/// but we want tight control over error messages (offsets included) and
/// zero-copy reads of null-terminated strings.
pub struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    pub fn offset(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    /// Skip `n` bytes. Useful for exploratory tooling that needs to
    /// jump to a specific offset before reading.
    pub fn advance(&mut self, n: usize) -> Result<()> {
        if self.pos + n > self.bytes.len() {
            anyhow::bail!(
                "advance by {n} past EOF (pos = {}, len = {})",
                self.pos,
                self.bytes.len()
            );
        }
        self.pos += n;
        Ok(())
    }

    /// Read a single byte.
    pub fn read_u8(&mut self) -> Result<u8> {
        let b = *self
            .bytes
            .get(self.pos)
            .with_context(|| format!("unexpected EOF reading u8 at offset {}", self.pos))?;
        self.pos += 1;
        Ok(b)
    }

    /// Read a little-endian `u32`.
    pub fn read_u32_le(&mut self) -> Result<u32> {
        let slice = self
            .bytes
            .get(self.pos..self.pos + 4)
            .with_context(|| format!("unexpected EOF reading u32 at offset {}", self.pos))?;
        let v = u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]);
        self.pos += 4;
        Ok(v)
    }

    /// Read a little-endian `f32`.
    pub fn read_f32_le(&mut self) -> Result<f32> {
        let bits = self.read_u32_le()?;
        Ok(f32::from_bits(bits))
    }

    /// View of the bytes the cursor hasn't consumed yet. Useful for
    /// scan-based section reads where we search for a known marker
    /// instead of parsing forward.
    pub fn bytes_after_cursor(&self) -> &'a [u8] {
        &self.bytes[self.pos..]
    }

    /// Read a null-terminated UTF-8 string, advancing past the `\0`.
    ///
    /// Returns a borrowed `&str` into the underlying buffer — callers
    /// that need ownership can `.to_owned()` explicitly.
    pub fn read_cstr(&mut self) -> Result<&'a str> {
        let start = self.pos;
        let end = self.bytes[start..]
            .iter()
            .position(|&b| b == 0)
            .map(|n| start + n)
            .with_context(|| {
                format!("unterminated C string starting at offset {start}")
            })?;
        let bytes = &self.bytes[start..end];
        let s = std::str::from_utf8(bytes).with_context(|| {
            format!(
                "invalid UTF-8 in C string at offset {start} (len {})",
                end - start
            )
        })?;
        self.pos = end + 1;
        Ok(s)
    }
}
