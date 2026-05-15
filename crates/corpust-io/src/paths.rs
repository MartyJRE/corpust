//! Platform-correct paths for persisted corpus data.
//!
//! Corpora live under a per-platform user data directory:
//!
//! ```text
//! <data_root>/
//! └── corpora/
//!     └── <slug>/
//!         ├── index/        ← tantivy files
//!         └── metadata.json ← versioned CorpusMeta sidecar
//! ```
//!
//! On macOS that's `~/Library/Application Support/corpust/`; on Linux
//! `$XDG_DATA_HOME/corpust/` (default `~/.local/share/corpust/`); on
//! Windows `%APPDATA%\corpust\`. We use [`BaseDirs`] + a single
//! `corpust/` join rather than [`ProjectDirs`] so the path stays
//! `corpust` on every platform — `ProjectDirs` would otherwise fold in
//! a qualifier/organization prefix.
//!
//! ## Override
//!
//! Set the `CORPUST_DATA_ROOT` environment variable to point the whole
//! tree somewhere else — typically an external drive. The value is
//! taken verbatim (no `corpust/` suffix appended), so a path like
//! `/Volumes/Big/corpust-data` will hold `corpora/<slug>/...` directly
//! under it.

use anyhow::{Result, anyhow};
use directories::BaseDirs;
use std::path::PathBuf;

/// Root of all corpust-owned user data for the current OS account.
///
/// Honors `$CORPUST_DATA_ROOT` if set; otherwise falls back to the
/// platform-specific data directory.
pub fn data_root() -> Result<PathBuf> {
    if let Ok(override_path) = std::env::var("CORPUST_DATA_ROOT")
        && !override_path.is_empty()
    {
        return Ok(PathBuf::from(override_path));
    }
    let base = BaseDirs::new().ok_or_else(|| anyhow!("no home directory available"))?;
    Ok(base.data_dir().join("corpust"))
}

/// Directory that holds every built corpus.
pub fn corpora_root() -> Result<PathBuf> {
    Ok(data_root()?.join("corpora"))
}

/// Directory for a corpus with the given slug.
pub fn corpus_dir(slug: &str) -> Result<PathBuf> {
    Ok(corpora_root()?.join(slug))
}

/// Path to the tantivy index inside a corpus dir.
pub fn index_path(slug: &str) -> Result<PathBuf> {
    Ok(corpus_dir(slug)?.join("index"))
}

/// Path to the metadata sidecar inside a corpus dir.
pub fn metadata_path(slug: &str) -> Result<PathBuf> {
    Ok(corpus_dir(slug)?.join("metadata.json"))
}

/// Turn a human-chosen corpus name into a filesystem-safe slug.
///
/// Lowercase ASCII, `[a-z0-9_-]` only, with runs of other characters
/// collapsed into a single dash. We keep it strict because the slug
/// becomes part of the on-disk path on three different OSes — being
/// forgiving about whitespace / accents isn't worth the cross-platform
/// edge cases.
pub fn slugify(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_dash = false;
    for c in name.chars() {
        let keep = if c.is_ascii_alphanumeric() {
            Some(c.to_ascii_lowercase())
        } else if c == '_' || c == '-' {
            Some(c)
        } else {
            None
        };
        match keep {
            Some(ch) => {
                out.push(ch);
                prev_dash = false;
            }
            None => {
                if !prev_dash && !out.is_empty() {
                    out.push('-');
                    prev_dash = true;
                }
            }
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        out.push_str("corpus");
    }
    out
}

/// Pick a slug that doesn't collide with an existing corpus directory.
///
/// If `<base>` is free we return it as-is. Otherwise we append
/// `-2`, `-3`, … until we find one that's free. Caller decides whether
/// to bail instead — we don't try to be clever about that here.
pub fn unique_slug(base: &str) -> Result<String> {
    let root = corpora_root()?;
    if !root.exists() {
        return Ok(base.to_owned());
    }
    if !root.join(base).exists() {
        return Ok(base.to_owned());
    }
    for n in 2u32..=9999 {
        let candidate = format!("{base}-{n}");
        if !root.join(&candidate).exists() {
            return Ok(candidate);
        }
    }
    Err(anyhow!("ran out of slug suffixes for base {base:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_is_strict_and_safe() {
        assert_eq!(slugify("Gutenberg · EN"), "gutenberg-en");
        assert_eq!(slugify("  spaces  "), "spaces");
        assert_eq!(slugify("a/b/c"), "a-b-c");
        assert_eq!(slugify("éclair"), "clair");
        assert_eq!(slugify("---"), "corpus");
        assert_eq!(slugify(""), "corpus");
        assert_eq!(slugify("my_corpus-1"), "my_corpus-1");
    }

    /// Combined to avoid a cargo-default-parallelism race on the
    /// shared `CORPUST_DATA_ROOT` env var: all env-driven paths are
    /// verified in one test.
    #[test]
    fn env_driven_paths_compose_under_override() {
        let prev = std::env::var("CORPUST_DATA_ROOT").ok();
        // SAFETY: setting/removing env vars is unsafe in Rust 2024
        // because other threads might read concurrently. This crate
        // only reads `CORPUST_DATA_ROOT` from `data_root()` and only
        // in this test, so the race window is bounded.
        unsafe {
            std::env::remove_var("CORPUST_DATA_ROOT");
        }
        let default_root = data_root().unwrap();
        assert!(
            default_root.ends_with("corpust"),
            "default root should end in `corpust`, got {default_root:?}"
        );

        let tmp = tempfile::tempdir().unwrap();
        let tmp_root = tmp.path().to_path_buf();
        unsafe {
            std::env::set_var("CORPUST_DATA_ROOT", &tmp_root);
        }

        assert_eq!(data_root().unwrap(), tmp_root);
        assert_eq!(corpora_root().unwrap(), tmp_root.join("corpora"));
        assert_eq!(corpus_dir("foo").unwrap(), tmp_root.join("corpora/foo"));
        assert_eq!(
            index_path("foo").unwrap(),
            tmp_root.join("corpora/foo/index")
        );
        assert_eq!(
            metadata_path("foo").unwrap(),
            tmp_root.join("corpora/foo/metadata.json")
        );

        // Empty CORPUST_DATA_ROOT falls back to the default lookup.
        unsafe {
            std::env::set_var("CORPUST_DATA_ROOT", "");
        }
        assert!(data_root().unwrap().ends_with("corpust"));

        // unique_slug: walks through `base`, `base-2`, … as dirs exist.
        unsafe {
            std::env::set_var("CORPUST_DATA_ROOT", &tmp_root);
        }
        // corpora root missing → returns the base verbatim.
        assert_eq!(unique_slug("fresh").unwrap(), "fresh");

        std::fs::create_dir_all(tmp_root.join("corpora")).unwrap();
        assert_eq!(unique_slug("fresh").unwrap(), "fresh");

        std::fs::create_dir(tmp_root.join("corpora/taken")).unwrap();
        assert_eq!(unique_slug("taken").unwrap(), "taken-2");

        std::fs::create_dir(tmp_root.join("corpora/taken-2")).unwrap();
        assert_eq!(unique_slug("taken").unwrap(), "taken-3");

        unsafe {
            match prev {
                Some(v) => std::env::set_var("CORPUST_DATA_ROOT", v),
                None => std::env::remove_var("CORPUST_DATA_ROOT"),
            }
        }
    }
}
