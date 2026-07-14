#!/usr/bin/env python3
"""Unit tests for scripts/report_rust_loc.py stripping heuristic and soft/strict exit.

Run:
  python3 scripts/test_report_rust_loc.py
  python3 -m unittest scripts.test_report_rust_loc
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Allow `python3 scripts/test_report_rust_loc.py` without package install.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import report_rust_loc as loc  # noqa: E402


class StripCfgTestModules(unittest.TestCase):
    def test_simple_strip(self) -> None:
        src = (
            "fn prod() {}\n"
            "\n"
            "#[cfg(test)]\n"
            "mod tests {\n"
            "    #[test]\n"
            "    fn t() {}\n"
            "}\n"
            "fn more() {}\n"
        )
        remaining, stripped, found = loc.strip_cfg_test_modules(src)
        self.assertTrue(found)
        self.assertGreater(stripped, 0)
        self.assertIn("fn prod()", remaining)
        self.assertIn("fn more()", remaining)
        self.assertNotIn("mod tests", remaining)
        self.assertNotIn("fn t()", remaining)

    def test_nested_braces_strip(self) -> None:
        src = (
            "fn prod() {}\n"
            "#[cfg(test)]\n"
            "mod tests {\n"
            "    fn inner() {\n"
            "        if true {\n"
            "            let _x = 1;\n"
            "        }\n"
            "    }\n"
            "    mod nested {\n"
            "        fn n() {}\n"
            "    }\n"
            "}\n"
            "fn after() {}\n"
        )
        remaining, stripped, found = loc.strip_cfg_test_modules(src)
        self.assertTrue(found)
        self.assertIn("fn prod()", remaining)
        self.assertIn("fn after()", remaining)
        self.assertNotIn("mod tests", remaining)
        self.assertNotIn("mod nested", remaining)
        self.assertEqual(remaining.count("{"), remaining.count("}"))

    def test_string_unbalanced_braces_still_strips(self) -> None:
        """Fixture strings with `{`/`}` must not prevent module close detection."""
        src = (
            "fn prod() {}\n"
            "#[cfg(test)]\n"
            "mod tests {\n"
            '    const BAD: &str = "{unclosed: brace\\n";\n'
            '    const YAML: &str = "key: {not: closed";\n'
            "    #[test]\n"
            "    fn t() {\n"
            '        let s = r#"{"raw": true"#;\n'
            "        let _ = s;\n"
            "    }\n"
            "}\n"
            "fn after() {}\n"
        )
        remaining, stripped, found = loc.strip_cfg_test_modules(src)
        self.assertTrue(found)
        self.assertGreater(stripped, 5)
        self.assertIn("fn prod()", remaining)
        self.assertIn("fn after()", remaining)
        self.assertNotIn("mod tests", remaining)
        self.assertNotIn("{unclosed", remaining)

    def test_comment_braces_ignored(self) -> None:
        src = (
            "fn prod() {}\n"
            "#[cfg(test)]\n"
            "mod tests {\n"
            "    // ignore { and } here\n"
            "    /* also { here } */\n"
            "    fn t() {}\n"
            "}\n"
        )
        remaining, stripped, found = loc.strip_cfg_test_modules(src)
        self.assertTrue(found)
        self.assertIn("fn prod()", remaining)
        self.assertNotIn("mod tests", remaining)

    def test_unbalanced_mod_left_alone(self) -> None:
        # Missing closing brace of mod — must not strip partially.
        src = (
            "fn prod() {}\n"
            "#[cfg(test)]\n"
            "mod tests {\n"
            "    fn t() {}\n"
            "fn after() {}\n"
        )
        remaining, stripped, found = loc.strip_cfg_test_modules(src)
        self.assertFalse(found)
        self.assertEqual(stripped, 0)
        self.assertEqual(remaining, src)

    def test_pub_crate_mod(self) -> None:
        src = (
            "fn prod() {}\n"
            "#[cfg(test)]\n"
            "pub(crate) mod tests {\n"
            "    fn t() {}\n"
            "}\n"
        )
        remaining, stripped, found = loc.strip_cfg_test_modules(src)
        self.assertTrue(found)
        self.assertNotIn("mod tests", remaining)

    def test_non_module_cfg_test_not_stripped(self) -> None:
        """#[cfg(test)] on free fn / items is intentionally left in PROD."""
        src = (
            "fn prod() {}\n"
            "#[cfg(test)]\n"
            "fn test_only_helper() {}\n"
            "fn more() {}\n"
        )
        remaining, stripped, found = loc.strip_cfg_test_modules(src)
        self.assertFalse(found)
        self.assertEqual(stripped, 0)
        self.assertIn("test_only_helper", remaining)

    def test_char_literal_braces(self) -> None:
        src = (
            "fn prod() {}\n"
            "#[cfg(test)]\n"
            "mod tests {\n"
            "    const A: char = '{';\n"
            "    const B: char = '}';\n"
            "    fn t() {}\n"
            "}\n"
            "fn after() {}\n"
        )
        remaining, stripped, found = loc.strip_cfg_test_modules(src)
        self.assertTrue(found)
        self.assertIn("fn after()", remaining)
        self.assertNotIn("mod tests", remaining)


class BucketAndExit(unittest.TestCase):
    def test_bucket_for(self) -> None:
        self.assertEqual(loc.bucket_for(100), ("ok", False))
        self.assertEqual(loc.bucket_for(loc.OK_MAX), ("ok", False))
        self.assertEqual(loc.bucket_for(loc.OK_MAX + 1), ("soft_over", False))
        self.assertEqual(loc.bucket_for(loc.SOFT_MAX), ("soft_over", False))
        self.assertEqual(loc.bucket_for(loc.SOFT_MAX + 1), ("warn", False))
        self.assertEqual(loc.bucket_for(loc.HARD_MAX + 1), ("warn", True))

    def test_soft_exit_zero_with_warns(self) -> None:
        """Soft mode always exits 0 even with oversized unallowlisted files."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            crate_src = root / "crates" / "demo" / "src"
            crate_src.mkdir(parents=True)
            # Build a file with production LOC > SOFT_MAX and no tests.
            body = "\n".join(f"fn f{i}() {{}}" for i in range(loc.SOFT_MAX + 50)) + "\n"
            (crate_src / "big.rs").write_text(body, encoding="utf-8")
            # Empty allowlist
            allow = root / "allow.txt"
            allow.write_text("", encoding="utf-8")
            code = loc.main(["--root", str(root), "--allowlist", str(allow)])
            self.assertEqual(code, 0)

    def test_strict_fails_non_allowlisted_warn(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            crate_src = root / "crates" / "demo" / "src"
            crate_src.mkdir(parents=True)
            body = "\n".join(f"fn f{i}() {{}}" for i in range(loc.SOFT_MAX + 50)) + "\n"
            (crate_src / "big.rs").write_text(body, encoding="utf-8")
            allow = root / "allow.txt"
            allow.write_text("", encoding="utf-8")
            code = loc.main(
                ["--root", str(root), "--allowlist", str(allow), "--strict"]
            )
            self.assertEqual(code, 1)

    def test_strict_ok_when_allowlisted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            crate_src = root / "crates" / "demo" / "src"
            crate_src.mkdir(parents=True)
            body = "\n".join(f"fn f{i}() {{}}" for i in range(loc.SOFT_MAX + 50)) + "\n"
            (crate_src / "big.rs").write_text(body, encoding="utf-8")
            allow = root / "allow.txt"
            allow.write_text("crates/demo/src/big.rs\n", encoding="utf-8")
            code = loc.main(
                ["--root", str(root), "--allowlist", str(allow), "--strict"]
            )
            self.assertEqual(code, 0)


class FindMatchingBrace(unittest.TestCase):
    def test_basic(self) -> None:
        s = "mod t { fn x() { } }"
        end = loc.find_matching_brace_end(s, s.index("{"))
        self.assertIsNotNone(end)
        self.assertEqual(s[end], "}")
        self.assertEqual(end, len(s) - 1)

    def test_string_brace(self) -> None:
        s = 'mod t { let s = "{"; }'
        end = loc.find_matching_brace_end(s, s.index("{"))
        self.assertEqual(end, len(s) - 1)


if __name__ == "__main__":
    unittest.main()
