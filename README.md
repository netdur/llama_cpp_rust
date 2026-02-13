# llama_cpp_rust

Rust workspace around `llama.cpp` with:

- `llama-cpp`: high-level Rust bindings
- `llama-cpp-ffi`: FFI layer and build integration
- `hugind_backend`: backend crate using the bindings

## Repository layout

This is a monorepo.  
`llama-cpp-ffi/vendor/llama.cpp` is tracked as a Git submodule.

## Build

```bash
cargo build
```

Metal build (macOS):

```bash
./build_metal.sh
```

## Vendor patch workflow

`llama-cpp-ffi/vendor/llama.cpp` stays close to upstream and local changes are
stored as patch files under `patches/llama.cpp/`.

After updating the submodule, re-apply local patches with:

```bash
./scripts/apply_vendor_patches.sh
```
