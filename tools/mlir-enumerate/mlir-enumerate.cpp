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
#include "GeneratorInfo.h"
#include "IRDLUtils.h"

#include "mlir/Conversion/PDLToPDLInterp/PDLToPDLInterp.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IRDLVerifiers.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
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

  static llvm::cl::opt<bool> count(
      "count",
      llvm::cl::desc(
          "Print the number of programs that would be generated, and halt"),
      llvm::cl::init(false));

  static llvm::cl::opt<int> minNumOps(
      "min-num-ops",
      llvm::cl::desc("Minimum number of non-constant operations"),
      llvm::cl::init(-1));

  // Number of non-constant operations to be printed.
  static llvm::cl::opt<int> maxNumOps(
      "max-num-ops",
      llvm::cl::desc("Maximum number of non-constant operations"),
      llvm::cl::init(3));

  // Number of non-constant operations to be printed.
  static llvm::cl::opt<bool> exactSize(
      "exact-size",
      llvm::cl::desc(
          "Should the generated programs have exactly max-num-ops operations"),
      llvm::cl::init(false));

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

  enum class ConstantKind { None, Constant, Synth };

  static llvm::cl::opt<ConstantKind> constantKind(
      "constant-kind", llvm::cl::desc("What kind of constants to generate"),
      llvm::cl::init(ConstantKind::Constant),
      llvm::cl::values(
          clEnumValN(ConstantKind::None, "none", "no constants"),
          clEnumValN(ConstantKind::Constant, "constant",
                     "Generate only the specified constants"),
          clEnumValN(
              ConstantKind::Synth, "synth",
              "Generate a synth.constant operation instead of constants")));

  static llvm::cl::opt<int> minConstantValue(
      "min-constant-value",
      llvm::cl::desc("Minimum value for integer constants"), llvm::cl::init(0));

  static llvm::cl::opt<int> maxConstantValue(
      "max-constant-value",
      llvm::cl::desc("Maximum value for integer constants"), llvm::cl::init(1));

  static llvm::cl::opt<int> seed(
      "seed", llvm::cl::desc("Specify random seed used in generation"),
      llvm::cl::init(-1));

  static llvm::cl::opt<bool> allowUnusedArguments(
      "allow-unused-arguments",
      llvm::cl::desc("Allow unused arguments in the generated functions"),
      llvm::cl::init(false));

  static llvm::cl::opt<std::string> buildingBlocks(
      "building-blocks",
      llvm::cl::desc("If provided and non-empty, construct new programs by "
                     "combining the programs from the provided file instead of "
                     "by using all the available operations."),
      llvm::cl::init(""));

  static llvm::cl::opt<std::string> excludeSubpatterns(
      "exclude-subpatterns",
      llvm::cl::desc("Do not emmit programs that contain patterns from the "
                     "provided file"),
      llvm::cl::init("/dev/null"));

  static llvm::cl::opt<bool> cse(
      "cse",
      llvm::cl::desc("Run CSE on the generated program before printing it"),
      llvm::cl::init(false));

  static llvm::cl::opt<Configuration> configuration(
      "configuration",
      llvm::cl::desc(
          "Configuration to use for generating types and attributes"),
      llvm::cl::init(Configuration::Arith),
      llvm::cl::values(
          clEnumValN(Configuration::Arith, "arith",
                     "Generate types and attributes for the arith "
                     "dialect (default)"),
          clEnumValN(Configuration::Comb, "comb",
                     "Generate types and attributes for the comb "
                     "dialect"),
          clEnumValN(Configuration::SMT, "smt",
                     "Generate types and attributes for the smt "
                     "dialect"),
          clEnumValN(Configuration::LLVM, "llvm",
                     "Generate types and attributes for the llvm "
                     "dialect"),
          clEnumValN(Configuration::Tensor, "tensor",
                     "Generate types and attributes for the tensor "
                     "dialect")));

  static llvm::cl::opt<std::string> bitVectorWidths(
      "smt-bitvector-widths",
      llvm::cl::desc("In case the configuration is set to \"smt\", this is a "
                     "list of comma-separated bitwidths. If not specified, "
                     "this corresponds to no BitVector instructions."),
      llvm::cl::init(""));

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR enumerator");

  MLIRContext ctx;
  ctx.allowUnregisteredDialects();

  // Printing flags
  OpPrintingFlags printingFlags;
  printingFlags.printGenericOpForm(printOpGeneric);

  std::vector<unsigned> smtBvWidths;
  {
    std::stringstream ss(bitVectorWidths);
    std::string width;
    while (std::getline(ss, width, ',')) {
      smtBvWidths.push_back(std::stoi(width));
    }
  }

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

  std::function<OwningOpRef<ModuleOp>(MLIRContext &, tree_guide::Chooser *,
                                      int)>
      programCreator;
  std::vector<OperationOp> availableOps = {};
  dialects->walk(
      [&availableOps](OperationOp op) { availableOps.push_back(op); });

  if (buildingBlocks.empty()) {
    // Get the list of operations we support.
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

    auto createValueOutOfThinAir =
        [&smtBvWidths, constantName](GeneratorInfo &info,
                                     Type type) -> std::optional<Value> {
      auto *ctx = info.builder.getContext();
      auto func = llvm::cast<mlir::func::FuncOp>(
          *info.builder.getInsertionBlock()->getParentOp());
      if (func.getNumArguments() < (unsigned int)info.maxNumArgs &&
          (constantKind == ConstantKind::None || info.chooser->choose(2) == 0))
        return info.addFunctionArgument(type);

      if (constantKind == ConstantKind::Synth) {
        auto *op = info.builder.create(UnknownLoc::get(ctx),
                                       StringAttr::get(ctx, "synth.constant"),
                                       {}, {type});
        return op->getResult(0);
      }

      if (configuration == Configuration::SMT &&
          mlir::isa<smt::BoolType>(type)) {
        bool value = (info.chooser->choose(2) == 0);
        auto op = info.builder.create<smt::BoolConstantOp>(UnknownLoc::get(ctx),
                                                           value);
        return op.getResult();
      }

      if (configuration == Configuration::SMT &&
          mlir::isa<smt::BitVectorType>(type)) {
        unsigned width = smtBvWidths[info.chooser->choose(smtBvWidths.size())];
        // Only enumerate 0 and 1 for now.
        uint64_t value = info.chooser->choose(2);
        auto op = info.builder.create<smt::BVConstantOp>(UnknownLoc::get(ctx),
                                                         value, width);
        return op.getResult();
      }

      if (constantKind == ConstantKind::Constant &&
          configuration == Configuration::Arith &&
          mlir::isa<mlir::IntegerType>(type)) {
        int64_t value =
            info.chooser->choose(maxConstantValue - minConstantValue + 1) +
            minConstantValue;
        auto valueAttr = IntegerAttr::get(type, value);
        auto typedValue = mlir::cast<TypedAttr>(valueAttr);
        auto constant = info.builder.create<arith::ConstantOp>(
            UnknownLoc::get(ctx), typedValue);
        return constant.getResult();
      }

      if (constantKind == ConstantKind::Constant &&
          configuration == Configuration::LLVM &&
          mlir::isa<mlir::IntegerType>(type)) {
        int64_t value =
            info.chooser->choose(maxConstantValue - minConstantValue + 1) +
            minConstantValue;
        auto valueAttr = IntegerAttr::get(type, value);
        auto typedValue = mlir::cast<TypedAttr>(valueAttr);
        auto constant = info.builder.create<LLVM::ConstantOp>(
            UnknownLoc::get(ctx), typedValue);
        return constant.getResult();
      }

      if (constantKind != ConstantKind::None && constantName != "") {
        if (auto intType = mlir::dyn_cast<IntegerType>(type)) {
          auto value =
              IntegerAttr::get(type, info.chooser->chooseUnimportant());

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

    if (minNumOps != -1) {
      if (minNumOps > maxNumOps) {
        llvm::errs()
            << "min-num-ops must be less than or equal to max-num-ops\n";
        return 1;
      }
      if (exactSize && minNumOps != maxNumOps) {
        llvm::errs() << "If exact-size is set, min-num-ops must be equal to "
                        "max-num-ops, or equal to -1\n";
        return 1;
      }
    }

    programCreator = [availableOps, smtBvWidths, createValueOutOfThinAir](
                         MLIRContext &ctx, tree_guide::Chooser *chooser,
                         int seed) {
      int currentMaxNumOps = maxNumOps;
      bool currentExactSize = exactSize;

      // In the case where we have a minimum and maximum number of operations,
      // we randomly pick a number in between and use an exact size generation.
      if (minNumOps != -1) {
        currentExactSize = true;
        std::random_device rd;
        std::uniform_int_distribution<int> dist(minNumOps, maxNumOps);
        currentMaxNumOps = dist(rd);
      }
      return createProgram(
          ctx, availableOps, getAvailableTypes(ctx, configuration, smtBvWidths),
          getAvailableAttributes(ctx, configuration), chooser, currentMaxNumOps,
          maxNumArgs, seed, createValueOutOfThinAir, currentExactSize);
    };
  } else {
    std::ifstream f(buildingBlocks);
    if (!f.is_open()) {
      llvm::errs() << "Unable to open file " << buildingBlocks << "\n";
      std::exit(1);
    }
    std::vector<std::vector<ModuleOp>> blocks;
    std::vector<ModuleOp> lastBlocks;
    std::string program;
    std::string line;
    while (std::getline(f, line)) {
      if (line == "// +++++") {
        if (lastBlocks.empty()) {
          llvm::errs() << "No building block for some size.\n";
          std::exit(1);
        }
        blocks.push_back(lastBlocks);
        lastBlocks = std::vector<ModuleOp>();
      } else if (line == "// -----") {
        auto config(&ctx);
        auto parsedModule = parseSourceString<ModuleOp>(program, config);
        if (!parsedModule) {
          llvm::errs() << "Unable to parse this illegal sub-pattern:\n"
                       << program;
          std::exit(1);
        }
        auto module = parsedModule.release();
        module.getOperation()->remove();
        lastBlocks.push_back(module);
        program = "";
      } else {
        program += line;
        program += "\n";
      }
    }

    programCreator = [=](MLIRContext &ctx, tree_guide::Chooser *chooser,
                         int seed) {
      return createProgramWithBuildingBlocks(
          ctx, availableOps, getAvailableTypes(ctx, configuration, smtBvWidths),
          getAvailableAttributes(ctx, configuration), blocks, chooser,
          maxNumOps, maxNumArgs, seed);
    };
  }

  // Get the list of illegal sub-patterns.
  RewritePatternSet illegals(&ctx);
  if (excludeSubpatterns != "/dev/null") {
    std::ifstream f(excludeSubpatterns);
    if (!f.is_open()) {
      llvm::errs() << "Unable to open file " << excludeSubpatterns << "\n";
      std::exit(1);
    }
    std::string pattern;
    std::string line;
    while (std::getline(f, line)) {
      if (line == "// -----") {
        auto config(&ctx);
        auto parsedModule = parseSourceString<ModuleOp>(pattern, config);
        if (!parsedModule) {
          llvm::errs() << "Unable to parse this illegal sub-pattern:\n"
                       << pattern;
          std::exit(1);
        }
        auto module = parsedModule.release();
        module.getOperation()->remove();

        PDLPatternModule pdlPattern(module);
        illegals.add(std::move(pdlPattern));
        pattern = "";
      } else {
        pattern += line;
        pattern += "\n";
      }
    }
  }
  FrozenRewritePatternSet frozenIllegals(
      std::forward<RewritePatternSet>(illegals));

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
    makeChooser = [guide{std::make_shared<tree_guide::EnumeratingGuide>()}]() {
      return guide->makeChooser();
    };
  }

  while (auto chooser = makeChooser()) {
    auto module = programCreator(ctx, chooser.get(), correctProgramCounter);

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

    // Optionally run CSE
    if (cse) {
      auto pm = PassManager::on<ModuleOp>(&ctx);
      pm.addPass(mlir::createCSEPass());
      if (failed(pm.run(*module))) {
        llvm::errs() << "Failed to run CSE on a generated program.\n";
        module->dump();
        continue;
      }
    }

    // Make sure the program does not contain an illegal subpattern.
    if (applyPatternsGreedily(module->getBodyRegion(), frozenIllegals,
                              {.maxIterations = 1, .maxNumRewrites = 1})
            .failed()) {
      // If there is a fail, that means we matched the pattern, and apply the
      // rewrite (which does nothing). Since the rewrite does nothing, the
      // pattern continues to match, which causes a failure.
      continue;
    }

    correctProgramCounter += 1;
    if (!count) {
      module->print(llvm::outs(), printingFlags);
      llvm::outs() << "// -----\n";
      llvm::outs().flush();
    }

    if (maxPrograms != -1 && correctProgramCounter >= (size_t)maxPrograms)
      break;

    if (pauseBetweenPrograms) {
      char c;
      std::cin >> c;
      if (c == 'q')
        break;
    }
  }

  if (count) {
    llvm::outs() << correctProgramCounter;
    llvm::outs() << "\n";
    llvm::outs().flush();
  }
}
