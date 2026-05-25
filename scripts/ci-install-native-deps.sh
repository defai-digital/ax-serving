#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "native CI dependencies are only required on macOS"
  exit 0
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "error: Homebrew is required to install native CI dependencies" >&2
  exit 1
fi

HOMEBREW_PREFIX="${HOMEBREW_PREFIX:-$(brew --prefix)}"

brew list protobuf >/dev/null 2>&1 || brew install protobuf
brew list mlx >/dev/null 2>&1 || brew install mlx
brew list mlx-c >/dev/null 2>&1 || brew install mlx-c

brew link --overwrite mlx || true
brew link --overwrite mlx-c || true

MLX_C_HEADER="${HOMEBREW_PREFIX}/include/mlx/c/array.h"
if [[ ! -f "${MLX_C_HEADER}" ]]; then
  echo "error: mlx-c header not found at ${MLX_C_HEADER}" >&2
  echo "Homebrew prefix: ${HOMEBREW_PREFIX}" >&2
  find "${HOMEBREW_PREFIX}/opt" -path '*/mlx/c/array.h' -print >&2 || true
  exit 1
fi

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "HOMEBREW_PREFIX=${HOMEBREW_PREFIX}"
    echo "CPATH=${HOMEBREW_PREFIX}/include"
    echo "LIBRARY_PATH=${HOMEBREW_PREFIX}/lib"
    echo "CXXFLAGS=-I${HOMEBREW_PREFIX}/include"
    echo "LDFLAGS=-L${HOMEBREW_PREFIX}/lib"
  } >> "${GITHUB_ENV}"
fi
