#!/usr/bin/env bash
set -euo pipefail

git submodule update --remote llama-cpp-ffi/vendor/llama.cpp
cargo build -p llama-cpp --features llama-cpp-ffi/metal

SHORT_HASH=$(git -C llama-cpp-ffi/vendor/llama.cpp rev-parse --short HEAD)
git add llama-cpp-ffi/vendor/llama.cpp
git commit -m "Update llama.cpp submodule to ${SHORT_HASH}"
git push
