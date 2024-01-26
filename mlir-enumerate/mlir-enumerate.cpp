//===- tblgen-extract.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <vector>

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
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace irdl;

/// Get the types that the fuzzer supports.
std::vector<Type> getAvailableTypes(MLIRContext &ctx) {
  Builder builder(&ctx);
  return {builder.getIntegerType(1), builder.getIntegerType(8),
          builder.getIntegerType(32)};
}

/// Get the types that the fuzzer supports.
std::vector<Attribute> getAvailableAttributes(MLIRContext &ctx) {
  Builder builder(&ctx);
  return {builder.getI64IntegerAttr(0), builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(2), builder.getI64IntegerAttr(3),
          builder.getI64IntegerAttr(4), builder.getI64IntegerAttr(5),
          builder.getI64IntegerAttr(6), builder.getI64IntegerAttr(7),
          builder.getI64IntegerAttr(8), builder.getI64IntegerAttr(9)};
}

/// Parse a file containing the dialects that we want to use.
std::optional<OwningOpRef<ModuleOp>>
parseIRDLDialects(MLIRContext &ctx, StringRef inputFilename) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return std::nullopt;
  }

  // Tell sourceMgr about this buffer, which is what the parser will pick
  // up.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());

  // Parse the IRDL file.
  bool wasThreadingEnabled = ctx.isMultithreadingEnabled();
  ctx.disableMultithreading();

  // Parse the input file and reset the context threading state.
  OwningOpRef<ModuleOp> module(parseSourceFile<ModuleOp>(sourceMgr, &ctx));
  ctx.enableMultithreading(wasThreadingEnabled);

  return module;
}

int main(int argc, char **argv) {

  // The IRDL file containing the dialects that we want to generate
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<IRDL file>"), llvm::cl::init("-"));

  // The IRDL file containing the dialects that we want to generate
  static llvm::cl::opt<std::string> outputFolder(
      "o", llvm::cl::desc("Output folder"), llvm::cl::init("-"));

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR enumerator");

  MLIRContext ctx;

  // Register all dialects
  DialectRegistry registry;
  registerAllDialects(registry);
  ctx.appendDialectRegistry(registry);
  ctx.getOrLoadDialect<irdl::IRDLDialect>();
  ctx.loadAllAvailableDialects();

  // Try to parse the dialects.
  auto optDialects = parseIRDLDialects(ctx, inputFilename);
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
  int seed = dist(rd);

  llvm::errs() << "seed " << seed << "\n";
  auto guide = tree_guide::DefaultGuide(seed);
  while (auto chooser = guide.makeChooser()) {
    auto module =
        createProgram(ctx, availableOps, getAvailableTypes(ctx),
                      getAvailableAttributes(ctx), chooser.get(), 100, 3, seed);
    if (!module)
      continue;
    programCounter += 1;
    // Some programs still won't verify, because IRDL is not expressive enough
    // to represent all constraints.
    // For now, we just discard those programs. Hopefully, there should be a
    // majority of programs that are verifying.
    {
      // We discard diagnostics here, so we don't print the errors of the
      // programs that are not verifying.
      ScopedDiagnosticHandler diagHandler(
          &ctx, [](Diagnostic &) { return success(); });
      if (verify(*module, true).failed())
        continue;
    }
    correctProgramCounter += 1;

    // Print the percentage of programs that are verifying
    // llvm::errs() << "Generated " << programCounter << " programs, "
    //              << (((float)correctProgramCounter / (float)programCounter)
    //              *
    //                  100.0f)
    //              << "% verifying \n\n\n";

    if (outputFolder == "-") {
      module->print(llvm::outs());
      break;
    }

    // Store the program in the output folder.
    std::string outputFilename = outputFolder + "/generated-program" +
                                 std::to_string(correctProgramCounter) +
                                 ".mlir";
    llvm::errs() << "Storing program in " << outputFilename << "\n";
    std::string errorMessage;
    auto outputFile = openOutputFile(outputFilename, &errorMessage);
    if (!outputFile) {
      llvm::errs() << "Error opening output file: " << errorMessage << "\n";
      return 1;
    }
    llvm::errs() << errorMessage << "\n";

    module->print(outputFile->os());
    outputFile->keep();
  }
}
