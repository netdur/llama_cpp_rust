#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULE_DIR="$ROOT_DIR/llama-cpp-ffi/vendor/llama.cpp"
PATCH_DIR="$ROOT_DIR/patches/llama.cpp"

if [[ ! -d "$SUBMODULE_DIR/.git" && ! -f "$SUBMODULE_DIR/.git" ]]; then
  echo "error: submodule not found at $SUBMODULE_DIR" >&2
  exit 1
fi

if [[ ! -d "$PATCH_DIR" ]]; then
  echo "No patches to apply."
  exit 0
fi

shopt -s nullglob
PATCHES=("$PATCH_DIR"/*.patch)
shopt -u nullglob

if [[ ${#PATCHES[@]} -eq 0 ]]; then
  echo "No patches to apply."
  exit 0
fi

for patch in "${PATCHES[@]}"; do
  echo "Applying $(basename "$patch")"
  git -C "$SUBMODULE_DIR" apply "$patch"
done

echo "Done. Review submodule changes with:"
echo "  git -C \"$SUBMODULE_DIR\" status --short"
