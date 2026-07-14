#!/usr/bin/env python3
"""
Report production Rust LOC under crates/**/src/**/*.rs with soft size thresholds.

Usage:
  python3 scripts/report_rust_loc.py
  python3 scripts/report_rust_loc.py --root .
  python3 scripts/report_rust_loc.py --allowlist scripts/loc_allowlist.txt
  python3 scripts/report_rust_loc.py --format markdown   # default: table
  python3 scripts/report_rust_loc.py --format tsv

Thresholds (production LOC after heuristic test stripping):
  ok         ≤  800   preferred ownership unit
  soft_over  ≤ 1500   acceptable for mature modules (over preferred unit)
  warn       > 1500   prefer split before large features
  hard       > 2500   hard review flag; avoid growing without a split plan

Soft mode (default): exit 0 always. Over-threshold files listed in
scripts/loc_allowlist.txt are marked allowlisted and do not block.

CI: non-blocking maintainability signal only. Do not hard-fail the workspace
on existing giants until decomposition PRs land.

#[cfg(test)] module stripping heuristic (best-effort, not a full AST):
  1. Strip contiguous blocks that start with a line matching `#[cfg(test)]`
     followed (optionally after blank/comment/`#[...]` attribute lines) by
     `mod <name> {` / `mod <name>{` through the matching closing brace at the
     same brace depth as the opening `mod` line.
  2. Do not claim accuracy for tests interleaved mid-function, for
     `#[cfg(test)]` on items other than modules, or for macro-generated modules.
  3. Counts are heuristic; re-measure with tokei/cloc if gating on exact numbers.

No third-party dependencies (Python 3 stdlib only). No runtime behavior change.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Bands (production LOC after stripping).
OK_MAX = 800
SOFT_MAX = 1500  # warn when production LOC > SOFT_MAX
HARD_MAX = 2500

DEFAULT_ALLOWLIST = "scripts/loc_allowlist.txt"

# `#[cfg(test)]` attribute line (allow whitespace / trailing comment).
_CFG_TEST_RE = re.compile(r"^\s*#\[cfg\s*\(\s*test\s*\)\]\s*(?://.*)?$")
# Optional extra attributes between cfg(test) and mod.
_ATTR_RE = re.compile(r"^\s*#\[")
# mod name {  (opening brace on same line — common style)
_MOD_OPEN_RE = re.compile(r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+(\w+)\s*\{")
# blank or line comment only
_BLANK_OR_COMMENT_RE = re.compile(r"^\s*(?://.*)?$")


@dataclass
class FileLoc:
    path: str  # repo-relative POSIX path
    gross: int
    production: int
    stripped: int
    bucket: str  # ok | soft_over | warn
    hard: bool
    allowlisted: bool
    had_cfg_test_mod: bool


def repo_root_from(start: Path) -> Path:
    """Prefer git top-level; fall back to parent of scripts/ or cwd."""
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / ".git").exists() and (p / "crates").is_dir():
            return p
    if (cur / "crates").is_dir():
        return cur
    if cur.name == "scripts" and (cur.parent / "crates").is_dir():
        return cur.parent
    return cur


def iter_production_rs(root: Path) -> list[Path]:
    """Walk crates/**/src/**/*.rs, excluding target/ and tests/ trees."""
    crates = root / "crates"
    if not crates.is_dir():
        return []
    out: list[Path] = []
    for path in crates.rglob("*.rs"):
        rel_parts = path.relative_to(root).parts
        if "target" in rel_parts:
            continue
        # Only production source: .../src/...
        if "src" not in rel_parts:
            continue
        # Skip paths under a tests directory (e.g. crates/foo/tests/...).
        if "tests" in rel_parts:
            continue
        # Require src segment after crate root (crates/<crate>/src/...)
        try:
            src_idx = rel_parts.index("src")
        except ValueError:
            continue
        if src_idx < 2:  # need crates/<name>/src
            continue
        out.append(path)
    return sorted(out)


def strip_cfg_test_modules(text: str) -> tuple[str, int, bool]:
    """
    Best-effort strip of `#[cfg(test)]` + `mod name { ... }` blocks.

    Returns (remaining_text, stripped_line_count, found_any).
    """
    lines = text.splitlines(keepends=True)
    n = len(lines)
    keep = [True] * n
    found = False
    i = 0
    while i < n:
        if not _CFG_TEST_RE.match(lines[i].rstrip("\n")):
            i += 1
            continue
        # Look ahead for mod { after blank/comment/attribute lines.
        j = i + 1
        while j < n and (
            _BLANK_OR_COMMENT_RE.match(lines[j].rstrip("\n"))
            or _ATTR_RE.match(lines[j])
        ):
            # Stop if we hit another cfg that isn't part of this item chain—
            # still allow attributes; blank/comment OK.
            j += 1
        if j >= n or not _MOD_OPEN_RE.match(lines[j].rstrip("\n")):
            i += 1
            continue
        # Brace match from the mod line.
        depth = 0
        k = j
        closed = False
        while k < n:
            # Count braces outside of strings is not perfect; good enough
            # for typical test modules.
            line = lines[k]
            for ch in line:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        closed = True
                        break
            if closed:
                break
            k += 1
        if not closed:
            # Unbalanced — leave alone.
            i += 1
            continue
        # Mark i..k inclusive for stripping.
        for t in range(i, k + 1):
            keep[t] = False
        found = True
        i = k + 1

    remaining_lines = [ln for t, ln in enumerate(lines) if keep[t]]
    stripped = sum(1 for t in range(n) if not keep[t])
    return "".join(remaining_lines), stripped, found


def count_lines(text: str) -> int:
    if not text:
        return 0
    # Match wc -l style: count newlines; trailing content without newline
    # still counts as a line if non-empty file.
    if text.endswith("\n"):
        return text.count("\n")
    return text.count("\n") + (1 if text else 0)


def bucket_for(prod: int) -> tuple[str, bool]:
    hard = prod > HARD_MAX
    if prod <= OK_MAX:
        return "ok", hard
    if prod <= SOFT_MAX:
        return "soft_over", hard
    return "warn", hard


def load_allowlist(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    entries: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Normalize to POSIX repo-relative.
        entries.add(line.replace("\\", "/").lstrip("./"))
    return entries


def analyze_file(path: Path, root: Path, allow: set[str]) -> FileLoc:
    rel = path.relative_to(root).as_posix()
    raw = path.read_text(encoding="utf-8", errors="replace")
    gross = count_lines(raw)
    remaining, stripped, had = strip_cfg_test_modules(raw)
    production = count_lines(remaining)
    bucket, hard = bucket_for(production)
    return FileLoc(
        path=rel,
        gross=gross,
        production=production,
        stripped=stripped,
        bucket=bucket,
        hard=hard,
        allowlisted=rel in allow,
        had_cfg_test_mod=had,
    )


def format_table(rows: list[FileLoc], *, only_issues: bool) -> str:
    show = rows
    if only_issues:
        show = [r for r in rows if r.bucket != "ok" or r.hard]
    if not show:
        return "(no files to list)\n"

    headers = ("PATH", "PROD", "GROSS", "STRIP", "BUCKET", "HARD", "ALLOW", "CFG_TEST")
    data = [
        (
            r.path,
            str(r.production),
            str(r.gross),
            str(r.stripped),
            r.bucket,
            "yes" if r.hard else "",
            "yes" if r.allowlisted else "",
            "yes" if r.had_cfg_test_mod else "",
        )
        for r in show
    ]
    widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cols: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    lines = [fmt_row(headers), fmt_row(tuple("-" * w for w in widths))]
    for row in data:
        lines.append(fmt_row(row))
    return "\n".join(lines) + "\n"


def format_tsv(rows: list[FileLoc]) -> str:
    lines = ["path\tproduction\tgross\tstripped\tbucket\thard\tallowlisted\thad_cfg_test_mod"]
    for r in rows:
        lines.append(
            "\t".join(
                [
                    r.path,
                    str(r.production),
                    str(r.gross),
                    str(r.stripped),
                    r.bucket,
                    "1" if r.hard else "0",
                    "1" if r.allowlisted else "0",
                    "1" if r.had_cfg_test_mod else "0",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def format_markdown(rows: list[FileLoc], *, only_issues: bool) -> str:
    show = rows
    if only_issues:
        show = [r for r in rows if r.bucket != "ok" or r.hard]
    lines = [
        "| path | prod | gross | strip | bucket | hard | allow | cfg_test |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for r in show:
        lines.append(
            f"| `{r.path}` | {r.production} | {r.gross} | {r.stripped} | "
            f"{r.bucket} | {'yes' if r.hard else ''} | "
            f"{'yes' if r.allowlisted else ''} | "
            f"{'yes' if r.had_cfg_test_mod else ''} |"
        )
    if len(lines) == 2:
        lines.append("| _(none)_ | | | | | | | |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report production Rust LOC with soft size thresholds."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root (default: detect from cwd / script location)",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help=f"Allowlist file (default: <root>/{DEFAULT_ALLOWLIST})",
    )
    parser.add_argument(
        "--format",
        choices=("table", "tsv", "markdown"),
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="List all files (default table/markdown show soft_over+warn only summary detail)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any non-allowlisted file is over SOFT_MAX (warn). Default: soft exit 0.",
    )
    args = parser.parse_args(argv)

    script_path = Path(__file__).resolve()
    root = args.root.resolve() if args.root else repo_root_from(Path.cwd())
    if not (root / "crates").is_dir():
        # Fall back via script location.
        root = repo_root_from(script_path.parent)

    allow_path = (
        args.allowlist
        if args.allowlist is not None
        else root / DEFAULT_ALLOWLIST
    )
    if args.allowlist is not None and not allow_path.is_absolute():
        allow_path = (Path.cwd() / allow_path).resolve()
    allow = load_allowlist(allow_path)

    files = iter_production_rs(root)
    rows = [analyze_file(p, root, allow) for p in files]
    # Sort: warn/hard first, then by production LOC desc.
    bucket_rank = {"warn": 0, "soft_over": 1, "ok": 2}
    rows.sort(key=lambda r: (bucket_rank.get(r.bucket, 9), -r.production, r.path))

    warn_rows = [r for r in rows if r.bucket == "warn"]
    hard_rows = [r for r in rows if r.hard]
    soft_rows = [r for r in rows if r.bucket == "soft_over"]
    unallowlisted_warn = [r for r in warn_rows if not r.allowlisted]

    print("Rust production LOC report")
    print(f"root: {root}")
    print(f"files: {len(rows)}")
    print(
        f"thresholds: ok≤{OK_MAX}  soft_over≤{SOFT_MAX}  warn>{SOFT_MAX}  hard>{HARD_MAX}"
    )
    print(f"allowlist: {allow_path} ({len(allow)} entries)")
    print(
        "heuristic: strips #[cfg(test)] mod { ... } blocks; "
        "counts are approximate (see script header)."
    )
    print()

    only_issues = not args.all and args.format != "tsv"
    if args.format == "tsv":
        sys.stdout.write(format_tsv(rows))
    elif args.format == "markdown":
        # Full summary table of issues by default.
        sys.stdout.write(format_markdown(rows, only_issues=only_issues))
    else:
        if only_issues:
            print(f"Files over preferred unit (prod > {OK_MAX}) or hard-flagged:")
            print()
        sys.stdout.write(format_table(rows, only_issues=only_issues))

    print()
    print("Summary")
    print(f"  ok:                    {sum(1 for r in rows if r.bucket == 'ok')}")
    print(f"  soft_over (≤{SOFT_MAX}):   {len(soft_rows)}")
    print(f"  warn (>{SOFT_MAX}):         {len(warn_rows)}")
    print(f"  hard (>{HARD_MAX}):         {len(hard_rows)}")
    print(f"  warn allowlisted:      {sum(1 for r in warn_rows if r.allowlisted)}")
    print(f"  warn not allowlisted:  {len(unallowlisted_warn)}")
    if warn_rows:
        print()
        print("Warn files (prod > 1500):")
        for r in warn_rows:
            tag = "allowlisted" if r.allowlisted else "NOT allowlisted"
            hard = " hard" if r.hard else ""
            print(f"  {r.production:5d}  {r.path}  [{tag}{hard}]")

    if args.strict and unallowlisted_warn:
        print()
        print(
            f"strict: {len(unallowlisted_warn)} non-allowlisted file(s) over "
            f"{SOFT_MAX} LOC — failing.",
            file=sys.stderr,
        )
        return 1

    # Soft mode: always exit 0 (even with warnings / unallowlisted giants).
    return 0


if __name__ == "__main__":
    sys.exit(main())
