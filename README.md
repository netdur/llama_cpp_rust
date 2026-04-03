# llama_cpp_rust

Rust workspace around `llama.cpp` with:

- `llama-cpp`: high-level Rust bindings
- `llama-cpp-ffi`: FFI layer and build integration

## Repository layout

This is a monorepo.  
`llama-cpp-ffi/vendor/llama.cpp` is tracked as a Git submodule.

## Build

Default (CPU + multimodal):

```bash
cargo build -p llama-cpp
```

With Metal (macOS GPU):

```bash
cargo build -p llama-cpp --features llama-cpp-ffi/metal
```

Release with Metal:

```bash
cargo build -p llama-cpp --release --features llama-cpp-ffi/metal
```

With CUDA (NVIDIA GPU):

```bash
cargo build -p llama-cpp --features llama-cpp-ffi/cuda
```

> `metal` and `cuda` are mutually exclusive. `mtmd` (multimodal) is enabled by default.

## Updating llama.cpp

To update the submodule to the latest upstream commit:

```bash
git submodule update --remote llama-cpp-ffi/vendor/llama.cpp
```

Then rebuild — bindgen regenerates the FFI bindings automatically from the headers:

```bash
cargo build -p llama-cpp
```

If everything compiles, the bindings are up to date. Commit the updated submodule pointer:

```bash
git add llama-cpp-ffi/vendor/llama.cpp
git commit -m "Update llama.cpp submodule"
```
