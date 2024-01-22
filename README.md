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
First create the generation dir: `mkdir generated`.


`./build/bin/mlir-enumerate dialects/arith.mlir -o generated` will run the
enumerator with the `arith` dialect, and generate the programs in the
`generated` folder.

## Testing MLIR code with xDSL-smt

The [`xdsl-smt`](https://github.com/opencompl/xdsl-smt) project has a
translation validation tool for `arith` that we can use to test random
programs. The `run_xdsl_tv.py` script will run the translation validation
tool on randomly generated programs.

```bash
# Install xdsl-smt in a virtual environment
cd PATH_TO_XDSL_SMT
python -m venv venv
source venv/bin/activate
pip install -e .

# Generate random programs and run the translation validation tool
cd PATH_TO_MLIR_FUZZ
python run_xdsl_tv.py
```

To test different passes, modify the `is-miscompiling.sh` script.