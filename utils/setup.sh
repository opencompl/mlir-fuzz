#!/bin/bash
mkdir -p build && cd build
export CMAKE_GENERATOR=Ninja
cmake -G Ninja ../ -DCMAKE_BUILD_TYPE=Debug \
                   -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
                   -DMLIR_DIR=$(pwd)/../dyn-dialect/llvm-project/build/lib/cmake/mlir \
                   -DLLVM_EXTERNAL_LIT=$(pwd)/../dyn-dialect/llvm-project/build/bin/llvm-lit
