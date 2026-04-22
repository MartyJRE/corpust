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
//! par-explore header <file>
//! par-explore hex    <file> --from <off> --len <bytes>
//! par-explore cstrs  <file> --from <off> --count <n>
//! par-explore u32s   <file> --from <off> --count <n>
//! par-explore f32s   <file> --from <off> --count <n>
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
        "" => {
            eprintln!("usage: par-explore <header|hex|cstrs|u32s|f32s> <file> [--from N] [--len N] [--count N]");
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
