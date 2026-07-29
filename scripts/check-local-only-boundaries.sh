#!/usr/bin/env bash
set -euo pipefail

readonly local_review_dir=".internal"

tracked_files="$(git ls-files -- "$local_review_dir")"
if [[ -n "$tracked_files" ]]; then
  echo "Local review files must not be tracked:" >&2
  echo "$tracked_files" >&2
  exit 1
fi

if ! git check-ignore -q -- "$local_review_dir/.privacy-sentinel"; then
  echo "The local review directory must remain ignored." >&2
  exit 1
fi

readonly public_reference_pattern='(^|[/[:space:](`"])(\.\./)*\.internal/'
if git grep -n -I -E "$public_reference_pattern" -- \
  ':(exclude).gitignore' \
  ':(exclude).dockerignore' \
  ':(exclude)scripts/check-local-only-boundaries.sh'
then
  echo "Public repository files must not link to local review material." >&2
  exit 1
fi

echo "Local-only repository boundaries are intact."
