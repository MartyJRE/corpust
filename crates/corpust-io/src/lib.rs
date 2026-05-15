//! Corpus ingestion.
//!
//! Phase 0 supports only `.txt` files in a directory tree. XML/TEI, docx, pdf
//! come later — each as a separate reader behind a common trait once we have
//! enough shape to know what the trait should look like.

pub mod metadata;
pub mod paths;

use anyhow::{Context, Result};
use corpust_core::{DocId, Document};
use std::path::Path;
use walkdir::WalkDir;

/// Recursively read every `.txt` file under `dir` and return them as documents
/// with monotonically assigned [`DocId`]s.
pub fn read_text_dir(dir: impl AsRef<Path>) -> Result<Vec<Document>> {
    let dir = dir.as_ref();
    let mut documents = Vec::new();

    for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("txt") {
            continue;
        }
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        documents.push(Document {
            id: documents.len() as DocId,
            path: path.to_path_buf(),
            text,
        });
    }

    Ok(documents)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn reads_txt_files_and_assigns_ids() {
        let dir = tempdir();
        fs::write(dir.path().join("a.txt"), "hello").unwrap();
        fs::write(dir.path().join("b.txt"), "world").unwrap();
        fs::write(dir.path().join("ignored.md"), "nope").unwrap();

        let mut docs = read_text_dir(dir.path()).unwrap();
        docs.sort_by_key(|d| d.path.clone());
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].text, "hello");
        assert_eq!(docs[1].text, "world");
        let ids: Vec<_> = docs.iter().map(|d| d.id).collect();
        assert!(ids.contains(&0) && ids.contains(&1));
    }

    fn tempdir() -> TempDir {
        let path = std::env::temp_dir().join(format!("corpust-io-test-{}", rand_suffix()));
        fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }

    fn rand_suffix() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}
