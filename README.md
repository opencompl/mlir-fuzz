# Fuzzing tools for MLIR

The project is highly experimental, and is not currently meant to be used.
It aims at deriving fuzzers, reducers, or mutators for MLIR dialects, using
IRDL and to avoid generating programs that do not verify.

## Building

- Setup LLVM and IRDL projects
  - `git submodule update --init --recursive`
  - `./utils/build-llvm.sh`
- Setup CMake
  - `./utils/setup.sh`
- Build the different tools (only mlir-enumerate for now)
  - `./utils/build.sh`

## Running

`./build/bin/mlir-enumerate dialects/arith.irdl -o generated` will run the
enumerator with the `arith` dialect, and generate the programs in the
`generated` folder.