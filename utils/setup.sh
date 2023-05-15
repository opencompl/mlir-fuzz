#!/bin/bash
mkdir -p build && cd build
export CMAKE_GENERATOR=Ninja
cmake -G Ninja ../ -DCMAKE_BUILD_TYPE=Release \
                   -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
                   -DMLIR_DIR=$(pwd)/../llvm-project/build/lib/cmake/mlir
