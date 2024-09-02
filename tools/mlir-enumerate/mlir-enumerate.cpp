//===- tblgen-extract.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cmath>
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

  static llvm::cl::opt<bool> printOpGeneric(
      "mlir-print-op-generic",
      llvm::cl::desc("Print the generic form of the operations"),
      llvm::cl::init(false));

  enum class Strategy { BFS, Random };

  static llvm::cl::opt<Strategy> strategy(
      "strategy", llvm::cl::desc("Strategy to use for enumeration"),
      llvm::cl::init(Strategy::BFS),
      llvm::cl::values(clEnumValN(Strategy::BFS, "bfs", "BFS strategy"),
                       clEnumValN(Strategy::Random, "random",
                                  "Random exploration (will not stop "
                                  "when all programs are generated)")));

  static llvm::cl::opt<int> maxPrograms(
      "max-programs",
      llvm::cl::desc(
          "Maximum number of verified programs to generate, -1 for infinite"),
      llvm::cl::init(-1));

  static llvm::cl::opt<bool> noConstants(
      "no-constants", llvm::cl::desc("Do not generate constants"),
      llvm::cl::init(false));

  static llvm::cl::opt<int> seed(
      "seed", llvm::cl::desc("Specify random seed used in generation"),
      llvm::cl::init(-1));

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR enumerator");

  MLIRContext ctx;
  ctx.allowUnregisteredDialects();

  // Printing flags
  OpPrintingFlags printingFlags;
  printingFlags.printGenericOpForm(printOpGeneric);

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

  StringRef constantName = "";
  dialects->walk([&constantName](DialectOp op) {
    if (op.getName() == "arith") {
      constantName = "arith.constant";
      return WalkResult::interrupt();
    }
    if (op.getName() == "comb") {
      constantName = "hw.constant";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  bool noConstantsBool = noConstants;
  auto createValueOutOfThinAir =
      [constantName, noConstantsBool](GeneratorInfo &info,
                                      Type type) -> std::optional<Value> {
    auto *ctx = info.builder.getContext();
    auto func = llvm::cast<mlir::func::FuncOp>(
        *info.builder.getInsertionBlock()->getParentOp());
    if (func.getNumArguments() < (unsigned int)info.maxNumArgs &&
        (noConstantsBool || info.chooser->choose(2) == 0))
      return info.addFunctionArgument(type);

    if (!noConstantsBool && constantName != "") {
      if (auto intType = type.dyn_cast<IntegerType>()) {
        auto value = IntegerAttr::get(type, info.chooser->chooseUnimportant());

        OperationState state(
            UnknownLoc::get(ctx), constantName, {}, {type},
            {NamedAttribute(StringAttr::get(ctx, "value"), value)});
        auto op = info.builder.create(state);
        return op->getResult(0);
      }
    }

    auto &domValues = info.dominatingValues[type];

    if (domValues.size()) {
      auto [value, valueIndex] = info.getValue(type);
      assert(value && "Error in generator logic");
      return *value;
    }

    return info.addFunctionArgument(type);
  };

  size_t programCounter = 0;
  size_t correctProgramCounter = 0;

  // set seed to a random positive integer
  if (seed == -1) {
    seed = std::abs((int)std::random_device{}());
  }

  // Create the correct guide depending on the chosen strategy
  std::function<std::unique_ptr<tree_guide::Chooser>()> makeChooser = nullptr;
  if (strategy == Strategy::Random) {
    makeChooser = [guide{std::make_shared<tree_guide::DefaultGuide>(seed)}]() {
      return guide->makeChooser();
    };
  } else if (strategy == Strategy::BFS) {
    makeChooser = [guide{std::make_shared<tree_guide::BFSGuide>(seed)}]() {
      return guide->makeChooser();
    };
  }

  while (auto chooser = makeChooser()) {
    auto module = createProgram(ctx, availableOps, getAvailableTypes(ctx),
                                getAvailableAttributes(ctx), chooser.get(),
                                maxNumOps, maxNumArgs, correctProgramCounter,
                                createValueOutOfThinAir);

    programCounter += 1;
    if (!module)
      continue;
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
    module->print(llvm::outs(), printingFlags);
    llvm::outs() << "// -----\n";
    llvm::outs().flush();

    if (maxPrograms != -1 && correctProgramCounter >= (size_t)maxPrograms)
      break;

    if (pauseBetweenPrograms) {
      char c;
      std::cin >> c;
      if (c == 'q')
        break;
    }
  }
}
