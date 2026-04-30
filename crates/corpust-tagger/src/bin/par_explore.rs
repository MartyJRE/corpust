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
//! par-explore dtree-walk-auto <file> [--count <n>]
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
        "dtree-walk-auto" => dump_dtree_walk_auto(&bytes, from, count),
        "" => {
            eprintln!(
                "usage: par-explore <header|hex|cstrs|u32s|f32s|dtree-walk|dtree-walk-auto> \
                 <file> [--from N] [--len N] [--count N]"
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
                     reserved={:>2}  back_pos_i={:>2}  test_tag={:>3} ({})",
                    n.reserved,
                    n.back_pos_i,
                    n.test_tag_id,
                    header.tag(n.test_tag_id).unwrap_or("?"),
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
    // Distribution of `back_pos_i` values across Internals — useful
    // to verify the model's `cl` (each internal carries one position
    // index in 0..cl). And `test_tag_id` range as a sanity check.
    use std::collections::BTreeMap;
    let mut by_back_pos: BTreeMap<u32, usize> = BTreeMap::new();
    let mut tt_min = u32::MAX;
    let mut tt_max = u32::MIN;
    for n in tree.internals() {
        *by_back_pos.entry(n.back_pos_i).or_insert(0) += 1;
        tt_min = tt_min.min(n.test_tag_id);
        tt_max = tt_max.max(n.test_tag_id);
    }
    print!("internals by back_pos_i: ");
    let parts: Vec<String> = by_back_pos
        .iter()
        .map(|(p, c)| format!("i={p} → {c}"))
        .collect();
    println!("{}", parts.join(", "));
    if !by_back_pos.is_empty() {
        println!("test_tag_id range: {tt_min}..={tt_max}");
    }
    Ok(())
}

/// Like `dtree-walk`, but locate the dtree section automatically by
/// scanning forward from where the lexicon ends. Then dump every
/// Internal record enriched with topology context — preorder index,
/// depth, yes-subtree size, no-subtree size — so we can spot any
/// correlation between `branch_info` and tree structure.
fn dump_dtree_walk_auto(bytes: &[u8], from: usize, count: usize) -> Result<()> {
    let mut cur = par::Cursor::new(bytes);
    let header = par::header::read(&mut cur)?;

    let dtree_start = if from > 0 {
        from
    } else {
        // Scan forward from the post-lexicon position. For toys this
        // works in milliseconds; for large files (english.par)
        // callers should pass --from with the known offset.
        let lexicon_end = match par::lexicon::read(&mut cur, &header) {
            Ok(_) => cur.offset(),
            Err(_) => cur.offset(),
        };
        par::dtree::find_dtree_start(bytes, &header, lexicon_end).with_context(|| {
            format!(
                "no offset in [{lexicon_end}, {}] parses as a dtree section",
                bytes.len()
            )
        })?
    };

    println!(
        "dtree at offset {dtree_start} (0x{dtree_start:x})"
    );

    let mut cur = par::Cursor::new(bytes);
    cur.advance(dtree_start)?;
    let tree = par::dtree::read(&mut cur, &header)?;
    let forest = tree.reconstruct()?;
    let counts = tree.kind_counts();
    println!(
        "{} records: {} internals, {} leaves, {} pruned-internals, {} defaults; \
         forest has {} tree(s), {} reconstructed nodes",
        tree.records.len(),
        counts.internals,
        counts.leaves,
        counts.pruned_internals,
        counts.defaults,
        forest.roots.len(),
        forest.nodes.len()
    );

    // Compute depth + subtree size per forest node.
    let mut depth = vec![0usize; forest.nodes.len()];
    let mut size = vec![1usize; forest.nodes.len()];
    for &root in &forest.roots {
        annotate(&forest.nodes, root, 0, &mut depth, &mut size);
    }

    // Walk forest.nodes in order — that's preorder DFS — and print
    // every Internal record with its predicate + topology context.
    // Predicate reads as `tag_at[-(i+1)] == test_tag`.
    println!();
    println!(
        "{:>6}  {:>5}  {:>3}  {:>3}  {:>11}  {:>4}  {:>4}",
        "fnidx", "depth", "i", "tt", "test_tag", "ysz", "nsz"
    );
    let mut shown = 0usize;
    let mut by_back_pos: std::collections::BTreeMap<u32, usize> =
        std::collections::BTreeMap::new();
    for (fnidx, node) in forest.nodes.iter().enumerate() {
        let par::dtree::TreeNode::Internal { predicate, yes, no } = node else {
            continue;
        };
        *by_back_pos.entry(predicate.back_pos_i).or_insert(0) += 1;
        if shown >= count {
            shown += 1;
            continue;
        }
        let tag_name = header.tag(predicate.test_tag_id).unwrap_or("?");
        println!(
            "{:>6}  {:>5}  {:>3}  {:>3}  {:>11}  {:>4}  {:>4}",
            fnidx,
            depth[fnidx],
            predicate.back_pos_i,
            predicate.test_tag_id,
            tag_name,
            size[*yes],
            size[*no],
        );
        shown += 1;
    }
    if shown > count {
        println!(
            "  … {} more internals (pass --count {} to see all)",
            shown - count,
            counts.internals
        );
    }

    println!();
    let parts: Vec<String> = by_back_pos
        .iter()
        .map(|(i, c)| format!("i={i} → {c}"))
        .collect();
    println!("predicate distribution: {}", parts.join(", "));
    Ok(())
}

fn annotate(
    nodes: &[par::dtree::TreeNode],
    idx: usize,
    d: usize,
    depth: &mut [usize],
    size: &mut [usize],
) -> usize {
    depth[idx] = d;
    let s = match &nodes[idx] {
        par::dtree::TreeNode::Leaf { .. } => 1,
        par::dtree::TreeNode::Internal { yes, no, .. } => {
            1 + annotate(nodes, *yes, d + 1, depth, size)
                + annotate(nodes, *no, d + 1, depth, size)
        }
    };
    size[idx] = s;
    s
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
