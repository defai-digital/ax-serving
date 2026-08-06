"""Validate reader-facing metadata in a built AX Serving wheel."""

from __future__ import annotations

import argparse
from email import policy
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()

    with ZipFile(args.wheel) as archive:
        metadata_paths = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise SystemExit(
                f"expected one .dist-info/METADATA entry, found {metadata_paths}"
            )
        metadata = BytesParser(policy=policy.default).parsebytes(
            archive.read(metadata_paths[0])
        )

    if metadata["Name"] != "ax-serving":
        raise SystemExit(f"unexpected package name: {metadata['Name']!r}")
    if metadata["Description-Content-Type"] != "text/markdown":
        raise SystemExit(
            "wheel must declare a Markdown long description, found "
            f"{metadata['Description-Content-Type']!r}"
        )

    description = metadata.get_payload().strip()
    if len(description) < 1_000:
        raise SystemExit("wheel long description is unexpectedly short")
    if not description.startswith("# AX Serving Python SDK"):
        raise SystemExit("wheel long description is missing the SDK heading")

    print(
        f"validated {metadata['Name']} {metadata['Version']} "
        f"with {len(description)} description characters"
    )


if __name__ == "__main__":
    main()
