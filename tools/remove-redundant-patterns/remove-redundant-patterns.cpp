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

  auto &mainModule = input.value();
  auto canonicalModule =
      cast<ModuleOp>((*mainModule)->getRegion(0).front().front());

  std::vector<FuncOp> canonicalFuncs;
  for (auto &op : canonicalModule.getRegion().front())
    canonicalFuncs.push_back(cast<FuncOp>(op.getRegion(0).front().front()));

  auto illegalModule =
      cast<ModuleOp>((*mainModule)->getRegion(0).front().back());
  std::vector<FuncOp> illegalFuncs;
  for (auto &op : illegalModule.getRegion().front())
    illegalFuncs.push_back(cast<FuncOp>(op.getRegion(0).front().front()));

  std::vector<bool> isIllegal(illegalFuncs.size(), false);

  // Remove illegal patterns that are sub-patterns of canonical patterns.
  for (size_t i = 0; i < illegalFuncs.size(); i++) {
    for (size_t j = 0; j < canonicalFuncs.size(); j++) {
      if (isSubPattern(illegalFuncs[i], canonicalFuncs[j])) {
        isIllegal[i] = true;
        break;
      }
    }
  }

  // Remove redundant illegal patterns.
  for (int i = illegalFuncs.size() - 1; i >= 0; i--) {
    if (isIllegal[i])
      continue;
    for (int j = 0; (size_t)j < illegalFuncs.size(); ++j) {
      if (isIllegal[j])
        continue;
      if (i == j)
        continue;
      if (isSubPattern(illegalFuncs[j], illegalFuncs[i])) {
        isIllegal[i] = true;
        break;
      }
    }
  }

  for (size_t i = 0; i < illegalFuncs.size(); i++) {
    if (isIllegal[i]) {
      llvm::outs() << "true" << "\n";
    } else {
      llvm::outs() << "false" << "\n";
    }
  }

  return 0;
}
