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

#[test]
fn index_without_name_falls_back_to_input_basename() {
    let data_root = tempdir().unwrap();
    let work = tempdir().unwrap();
    let input = work.path().join("my-folder");
    fs::create_dir(&input).unwrap();
    fs::write(input.join("a.txt"), "hello world\n").unwrap();

    // No --out, no --name: slug should derive from "my-folder" and
    // land under CORPUST_DATA_ROOT/corpora/my-folder/.
    corpust()
        .env("CORPUST_DATA_ROOT", data_root.path())
        .args(["index"])
        .arg(&input)
        .assert()
        .success()
        .stdout(contains("indexed 1 doc"));

    let landed = data_root.path().join("corpora/my-folder/index");
    assert!(landed.exists(), "expected {landed:?} to exist");
    let meta = data_root.path().join("corpora/my-folder/metadata.json");
    assert!(meta.exists(), "expected {meta:?} to exist");
}

#[test]
fn index_with_annotate_writes_lemma_and_pos() {
    let work = tempdir().unwrap();
    let input = work.path().join("docs");
    fs::create_dir(&input).unwrap();
    fs::write(input.join("a.txt"), "The cat sat on the mat.\n").unwrap();
    let out = work.path().join("idx");

    // resources/treetagger lives at the workspace root; tests run from
    // the crate dir, so step two parents up to find it.
    let bundle = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("resources/treetagger");
    if !bundle.exists() {
        // Bundle not vendored in this checkout — skip.
        return;
    }

    corpust()
        .args(["index"])
        .arg(&input)
        .args(["--name", "annotated"])
        .args(["--out"])
        .arg(&out)
        .args(["--annotate"])
        .args(["--tagger-bundle"])
        .arg(&bundle)
        .assert()
        .success()
        .stdout(contains("annotation enabled"))
        .stdout(contains("indexed 1 doc"));

    // KWIC by lemma and POS — both go through the LayerArg conversion
    // arms that were uncovered.
    corpust()
        .args(["kwic", "--index"])
        .arg(&out)
        .arg("sit")
        .args(["--layer", "lemma"])
        .assert()
        .success();

    corpust()
        .args(["kwic", "--index"])
        .arg(&out)
        .arg("NN")
        .args(["--layer", "pos"])
        .assert()
        .success();
}
