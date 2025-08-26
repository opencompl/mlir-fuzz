//===- tblgen-extract.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <fstream>
#include <vector>

#include "CLITool.h"
#include "SubPattern.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace mlir::func;

int main(int argc, char **argv) {

  // The input file containing the two module we want to compare
  static llvm::cl::opt<std::string> inputFile(llvm::cl::Positional,
                                              llvm::cl::desc("<input file>"),
                                              llvm::cl::init("-"));

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR enumerator");

  MLIRContext ctx;
  ctx.allowUnregisteredDialects();

  // Register all dialects
  DialectRegistry registry;
  registerAllDialects(registry);
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();

  // Try to parse the dialects.
  auto input = parseMLIRFile(ctx, inputFile);
  if (!input)
    return 1;

  auto &module = input.value();
  ModuleOp lhsModule = cast<ModuleOp>((*module)->getRegion(0).front().front());
  FuncOp lhsFunc = cast<FuncOp>(lhsModule->getRegion(0).front().front());
  ModuleOp rhsModule = cast<ModuleOp>(*lhsModule->getNextNode());
  FuncOp rhsFunc = cast<FuncOp>(rhsModule->getRegion(0).front().front());

  if (isSubPattern(lhsFunc, rhsFunc)) {
    llvm::outs() << "true";
  } else {
    llvm::outs() << "false";
  }

  return 0;
}
