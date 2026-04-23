//! `.par` format exploration helper.
//!
//! Not user-facing. This binary exists to iteratively reverse-engineer
//! the TreeTagger parameter-file layout. Each subcommand dumps a view
//! of the file — header fields, raw hex at an offset, runs of
//! null-terminated strings, little-endian `u32` / `f32` streams — so we
//! can see what lives where without writing throw-away code every time.
//!
//! Usage:
//!
//! ```text
//! par-explore header     <file>
//! par-explore hex        <file> --from <off> --len <bytes>
//! par-explore cstrs      <file> --from <off> --count <n>
//! par-explore u32s       <file> --from <off> --count <n>
//! par-explore f32s       <file> --from <off> --count <n>
//! par-explore dtree-walk <file> --from <off> [--count <n>]
//! ```
//!
//! Offsets and lengths are decimal by default, `0x...` parses as hex.

use anyhow::{Context, Result, bail};
use corpust_tagger::par;
use std::env;
use std::fmt::Write as _;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut it = args.iter();
    let cmd = it.next().map(String::as_str).unwrap_or("");
    let file: PathBuf = it
        .next()
        .map(PathBuf::from)
        .context("missing <file> argument")?;

    let mut from: usize = 0;
    let mut len: usize = 128;
    let mut count: usize = 16;

    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--from" => from = parse_num(it.next().context("--from needs a value")?)?,
            "--len" => len = parse_num(it.next().context("--len needs a value")?)?,
            "--count" => count = parse_num(it.next().context("--count needs a value")?)?,
            other => bail!("unknown flag {other}"),
        }
    }

    let bytes = std::fs::read(&file).with_context(|| format!("reading {}", file.display()))?;

    match cmd {
        "header" => dump_header(&bytes),
        "hex" => dump_hex(&bytes, from, len),
        "cstrs" => dump_cstrs(&bytes, from, count),
        "u32s" => dump_u32s(&bytes, from, count),
        "f32s" => dump_f32s(&bytes, from, count),
        "dtree-walk" => dump_dtree_walk(&bytes, from, count),
        "" => {
            eprintln!(
                "usage: par-explore <header|hex|cstrs|u32s|f32s|dtree-walk> <file> \
                 [--from N] [--len N] [--count N]"
            );
            std::process::exit(2);
        }
        other => bail!("unknown subcommand {other}"),
    }
}

fn parse_num(s: &str) -> Result<usize> {
    if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Ok(usize::from_str_radix(h, 16)?)
    } else {
        Ok(s.parse()?)
    }
}

fn dump_header(bytes: &[u8]) -> Result<()> {
    let mut cur = par::Cursor::new(bytes);
    let header = par::header::read(&mut cur)?;
    println!("field_a        = {}", header.field_a);
    println!("field_b        = {}", header.field_b);
    println!("sent_tag_index = {}", header.sent_tag_index);
    println!("num_tags       = {}", header.tags.len());
    println!("tags:");
    for (i, tag) in header.tags.iter().enumerate() {
        println!("  [{i:>3}] {tag}");
    }
    println!();
    println!("section ends at offset {} (0x{:x})", header.end_offset, header.end_offset);
    println!("file size = {} bytes", bytes.len());
    Ok(())
}

fn dump_hex(bytes: &[u8], from: usize, len: usize) -> Result<()> {
    let end = (from + len).min(bytes.len());
    let slice = &bytes[from..end];
    for (row, chunk) in slice.chunks(16).enumerate() {
        let mut line = format!("{:08x}  ", from + row * 16);
        for (i, b) in chunk.iter().enumerate() {
            write!(line, "{b:02x}").unwrap();
            line.push(if i == 7 { '-' } else { ' ' });
        }
        for _ in chunk.len()..16 {
            line.push_str("   ");
        }
        line.push_str(" |");
        for b in chunk {
            line.push(if (0x20..0x7f).contains(b) { *b as char } else { '.' });
        }
        line.push('|');
        println!("{line}");
    }
    Ok(())
}

fn dump_cstrs(bytes: &[u8], from: usize, count: usize) -> Result<()> {
    let mut cur = par::Cursor::new(bytes);
    cur.advance(from)?;
    let mut printed = 0;
    while printed < count && cur.remaining() > 0 {
        let off = cur.offset();
        match cur.read_cstr() {
            Ok(s) => {
                println!("0x{off:08x}  [{:>4}b]  {s:?}", s.len());
                printed += 1;
            }
            Err(e) => {
                println!("0x{off:08x}  <stop: {e}>");
                break;
            }
        }
    }
    println!();
    println!("next offset = 0x{:x} ({})", cur.offset(), cur.offset());
    Ok(())
}

fn dump_u32s(bytes: &[u8], from: usize, count: usize) -> Result<()> {
    for i in 0..count {
        let off = from + i * 4;
        if off + 4 > bytes.len() {
            break;
        }
        let v = u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        let vi = v as i32;
        println!(
            "0x{off:08x}  u32 LE = {v:>12}  (i32 = {vi:>12}, hex = 0x{v:08x})"
        );
    }
    Ok(())
}

/// Walk the decision-tree section from `from` and print each record's
/// kind + key fields in order. Use `--count` to cap output (default
/// 16). The total record counts per kind are printed at the end so
/// you can spot-check them against `train-tree-tagger`'s reported
/// "Number of nodes: K".
fn dump_dtree_walk(bytes: &[u8], from: usize, count: usize) -> Result<()> {
    let mut cur = par::Cursor::new(bytes);
    let header = par::header::read(&mut cur)?;
    // header::read leaves the cursor just after the tag table, not at
    // the dtree — reset and advance to the caller-supplied offset.
    let mut cur = par::Cursor::new(bytes);
    cur.advance(from)?;
    let tree = par::dtree::read(&mut cur, &header)?;

    println!(
        "dtree @ 0x{from:x} ({from}): {} records total",
        tree.records.len()
    );
    let mut section_off = 0usize;
    for (i, rec) in tree.records.iter().enumerate() {
        if i >= count {
            println!("  … (truncated; pass --count {} to see all)", tree.records.len());
            break;
        }
        let file_off = from + section_off;
        match rec {
            par::dtree::DTreeRecord::Internal(n) => {
                println!(
                    "  [{i:>5}] @ sec+{section_off:>7} / 0x{file_off:08x}  Internal  \
                     offset_i={:>2}  tag_id={:>3} ({})  branch_info={}",
                    n.offset_i,
                    n.tag_id,
                    header.tag(n.tag_id).unwrap_or("?"),
                    n.branch_info
                );
                section_off += 12;
            }
            par::dtree::DTreeRecord::Leaf(l) => {
                println!(
                    "  [{i:>5}] @ sec+{section_off:>7} / 0x{file_off:08x}  Leaf     \
                     node_id={:>6}  weight={:>7}",
                    l.node_id, l.distribution.weight
                );
                section_off += 16 + header.tags.len() * 12;
            }
            par::dtree::DTreeRecord::PrunedInternal(p) => {
                println!(
                    "  [{i:>5}] @ sec+{section_off:>7} / 0x{file_off:08x}  PrunedInt \
                     weight={:>7}",
                    p.distribution.weight
                );
                section_off += 12 + header.tags.len() * 12;
            }
            par::dtree::DTreeRecord::Default(d) => {
                println!(
                    "  [{i:>5}] @ sec+{section_off:>7} / 0x{file_off:08x}  Default  \
                     weight={:>7}  (EOF)",
                    d.distribution.weight
                );
                section_off += 12 + header.tags.len() * 12;
            }
        }
    }
    let c = tree.kind_counts();
    println!();
    println!(
        "summary: {} internals, {} leaves, {} pruned-internals, {} defaults",
        c.internals, c.leaves, c.pruned_internals, c.defaults
    );
    // Break down the 12-byte Internal records by their first two
    // u32s — sub-task 2 needs to know how `offset_i` and `tag_id`
    // co-vary (the plan's toy models saw `offset_i ∈ {1,2,3}` but
    // `english.par` shows a different pattern), plus `branch_info`'s
    // range, to pick which corpus to train a toy against.
    use std::collections::BTreeMap;
    let mut by_header: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    let mut branch_min = u32::MAX;
    let mut branch_max = u32::MIN;
    for n in tree.internals() {
        *by_header.entry((n.offset_i, n.tag_id)).or_insert(0) += 1;
        branch_min = branch_min.min(n.branch_info);
        branch_max = branch_max.max(n.branch_info);
    }
    print!("internals by (offset_i, tag_id): ");
    let parts: Vec<String> = by_header
        .iter()
        .map(|((o, t), c)| format!("({o},{t})={c}"))
        .collect();
    println!("{}", parts.join(", "));
    if !by_header.is_empty() {
        println!("branch_info range: {branch_min}..={branch_max}");
    }
    Ok(())
}

fn dump_f32s(bytes: &[u8], from: usize, count: usize) -> Result<()> {
    for i in 0..count {
        let off = from + i * 4;
        if off + 4 > bytes.len() {
            break;
        }
        let v = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        println!("0x{off:08x}  f32 LE = {v:>20}");
    }
    Ok(())
}
