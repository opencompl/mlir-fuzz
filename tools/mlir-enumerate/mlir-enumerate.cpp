//===- tblgen-extract.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "CLITool.h"
#include "GeneratorInfo.h"
#include "IRDLUtils.h"

#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IRDLVerifiers.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace irdl;

int main(int argc, char **argv) {

  // The IRDL file containing the dialects that we want to generate
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<IRDL file>"), llvm::cl::init("-"));

  // Expect a new line before printing the next program.
  static llvm::cl::opt<bool> pauseBetweenPrograms(
      "pause-between-programs",
      llvm::cl::desc(
          "Expect a new line in stdin before printing the next program"),
      llvm::cl::init(false));

  // Number of non-constant operations to be printed.
  static llvm::cl::opt<int> maxNumOps(
      "max-num-ops",
      llvm::cl::desc("Maximum number of non-constant operations"),
      llvm::cl::init(3));

  // Maximum number of arguments to be added per function.
  static llvm::cl::opt<int> maxNumArgs(
      "max-num-args",
      llvm::cl::desc("Maximum number of arguments per function"),
      llvm::cl::init(3));

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
  auto optDialects = parseMLIRFile(ctx, inputFilename);
  if (!optDialects)
    return 1;

  // Get the dialects.
  auto &dialects = optDialects.value();

  // Get the list of operations we support.
  std::vector<OperationOp> availableOps = {};
  dialects->walk(
      [&availableOps](OperationOp op) { availableOps.push_back(op); });

  size_t programCounter = 0;
  size_t correctProgramCounter = 0;

  std::random_device rd;
  std::uniform_int_distribution<int> dist(0, 1 << 30);

  auto guide = tree_guide::BFSGuide();
  while (auto chooser = guide.makeChooser()) {
    auto module = createProgram(ctx, availableOps, getAvailableTypes(ctx),
                                getAvailableAttributes(ctx), chooser.get(),
                                maxNumOps, maxNumArgs, correctProgramCounter);

    programCounter += 1;
    // Some programs still won't verify, because IRDL is not expressive enough
    // to represent all constraints.
    {
      // We discard diagnostics here, so we don't print the errors of the
      // programs that are not verifying.
      ScopedDiagnosticHandler diagHandler(
          &ctx, [](Diagnostic &) { return success(); });
      if (verify(*module, true).failed())
        continue;
    }
    correctProgramCounter += 1;

    // Print the program to stdout.
    module->print(llvm::outs());
    llvm::outs().flush();

    if (pauseBetweenPrograms) {
      char c;
      std::cin >> c;
      if (c == 'q')
        break;
    }
  }

  llvm::outs() << "Generated " << programCounter << " programs, "
               << correctProgramCounter << " of which verify.\n";
}
