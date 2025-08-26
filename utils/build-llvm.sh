#!/usr/bin/env bash

BUILD_DIR=${1:-"build"}
INSTALL_DIR=${2:-"install"}

if command -v mold > /dev/null; then
  BEST_LINKER="mold"
elif command -v lld > /dev/null; then
  BEST_LINKER="lld"
else
  BEST_LINKER="ld"
fi

LINKER=${3:-${BEST_LINKER}}

mkdir -p llvm-project/$BUILD_DIR
mkdir -p llvm-project/$INSTALL_DIR
cd llvm-project/$BUILD_DIR
cmake -G Ninja ../llvm \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_INSTALL_PREFIX=../$INSTALL_DIR \
  -DLLVM_ENABLE_PROJECTS='mlir' \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_ENABLE_OCAMLDOC=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_INSTALL_UTILS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_USE_LINKER=${LINKER} \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_PDL_IN_PATTERNMATCH=ON \
  -DBUILD_SHARED_LIBS=ON

ninja install
