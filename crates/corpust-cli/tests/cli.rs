//! End-to-end coverage of the `corpust` binary.
//!
//! The annotation-heavy paths (`--annotate`, `annotate install-lang`) are
//! deliberately skipped — they touch real TreeTagger assets or the network.
//! The smoke flows below still drive the CLI through arg parsing, the index
//! build pipeline, the metadata sidecar writer, and the KWIC path.

use assert_cmd::Command;
use predicates::str::contains;
use std::fs;
use tempfile::tempdir;

fn corpust() -> Command {
    Command::cargo_bin("corpust").expect("corpust binary built")
}

#[test]
fn top_level_help_lists_subcommands() {
    corpust()
        .arg("--help")
        .assert()
        .success()
        .stdout(contains("index"))
        .stdout(contains("kwic"))
        .stdout(contains("annotate"));
}

#[test]
fn version_flag_prints_workspace_version() {
    corpust()
        .arg("--version")
        .assert()
        .success()
        .stdout(contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn index_then_kwic_word_layer() {
    let work = tempdir().unwrap();
    let input = work.path().join("docs");
    fs::create_dir(&input).unwrap();
    fs::write(
        input.join("a.txt"),
        "the quick brown fox jumps over the lazy dog\n",
    )
    .unwrap();
    fs::write(input.join("b.txt"), "the rain in spain stays mainly\n").unwrap();

    let out = work.path().join("idx");
    corpust()
        .args(["index"])
        .arg(&input)
        .args(["--name", "smoke"])
        .args(["--out"])
        .arg(&out)
        .assert()
        .success()
        .stdout(contains("indexed 2 doc"));

    // metadata sidecar should sit next to the index, not inside it.
    let meta_path = out.parent().unwrap().join("metadata.json");
    assert!(
        meta_path.exists(),
        "expected metadata.json at {meta_path:?}"
    );
    let meta_raw = fs::read_to_string(&meta_path).unwrap();
    assert!(meta_raw.contains("\"schemaVersion\""));
    assert!(meta_raw.contains("\"docCount\": 2"));

    corpust()
        .args(["kwic"])
        .args(["--index"])
        .arg(&out)
        .arg("the")
        .assert()
        .success()
        .stdout(contains("the"));
}

#[test]
fn kwic_against_missing_index_errors() {
    let work = tempdir().unwrap();
    corpust()
        .args(["kwic", "--index"])
        .arg(work.path().join("nope"))
        .arg("anything")
        .assert()
        .failure();
}

#[test]
fn install_lang_rejects_unknown_code() {
    corpust()
        .args(["annotate", "install-lang", "zz"])
        .assert()
        .failure()
        .stderr(contains("zz"));
}
