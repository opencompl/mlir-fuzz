//===- superoptimizer.cpp ---------------------------------------*- C++ -*-===//
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
#include "mlir/Dialect/SMT/IR/SMTTypes.h"
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

Type convertTypeToConfiguration(Type type, Configuration config) {
  switch (config) {
  case Configuration::Arith:
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
      return intType;
    if (auto smtBvType = mlir::dyn_cast<smt::BitVectorType>(type))
      return mlir::IntegerType::get(type.getContext(), smtBvType.getWidth());
    break;
  default:
    break;
  }
  llvm::errs() << "Unsupported type conversion from " << type
               << " to provided configuration.\n";
  return type;
}

func::FuncOp cloneFunctionWithConfiguration(func::FuncOp func,
                                            Configuration config,
                                            OpBuilder &builder) {
  llvm::SmallVector<Type> input_types;
  for (unsigned int i = 0; i < func.getFunctionType().getNumInputs(); i++) {
    input_types.push_back(
        convertTypeToConfiguration(func.getFunctionType().getInput(i), config));
  }
  llvm::SmallVector<Type> output_types;
  for (unsigned int i = 0; i < func.getFunctionType().getNumResults(); i++) {
    output_types.push_back(convertTypeToConfiguration(
        func.getFunctionType().getResult(i), config));
  }
  auto funcType =
      FunctionType::get(func.getContext(), input_types, output_types);
  auto newFunc = func.cloneWithoutRegions();
  newFunc.setType(funcType);
  builder.insert(newFunc);
  return newFunc;
}

/// Create a random program, given the decisions taken from chooser.
/// The program has at most `fuel` operations.
OwningOpRef<ModuleOp> createProgramFromInput(
    MLIRContext &ctx, func::FuncOp inputFunction,
    ArrayRef<OperationOp> availableOps, ArrayRef<Type> availableTypes,
    ArrayRef<Attribute> availableAttributes, tree_guide::Chooser *chooser,
    int numOps,
    GeneratorInfo::CreateValueOutOfThinAirFn createValueOutOfThinAir,
    bool useInputOps, Configuration config) {
  // Create an empty module.
  auto unknownLoc = UnknownLoc::get(&ctx);
  OwningOpRef<ModuleOp> module(ModuleOp::create(unknownLoc));

  // Create the builder, and set its insertion point in the module.
  OpBuilder builder(&ctx);
  auto &moduleBlock = module->getRegion().getBlocks().front();
  builder.setInsertionPoint(&moduleBlock, moduleBlock.begin());

  // Clone the input function
  func::FuncOp func;
  if (useInputOps) {
    auto funcOp = builder.insert(inputFunction.clone());
    func = cast<func::FuncOp>(funcOp);
    // We delete the return function, as well as the root operation we want to
    // replace.
    func.getBlocks().front().getTerminator()->erase();
    func.getBlocks().front().back().erase();
  } else {
    func = cloneFunctionWithConfiguration(inputFunction, config, builder);
    func.addEntryBlock();
  }

  // Set the insertion point to it
  auto &funcBlock = func.getRegion().front();
  builder.setInsertionPoint(&funcBlock, funcBlock.end());

  // Create the generator info
  GeneratorInfo info(chooser, builder, availableOps, availableTypes,
                     availableAttributes, 0, createValueOutOfThinAir);

  // Add all the function values to the dominating values
  for (auto arg : func.getArguments())
    info.addDominatingValue(arg);
  for (auto &op : func.getOps())
    for (auto result : op.getResults())
      info.addDominatingValue(result);

  auto type = func.getFunctionType().getResult(0);
  auto [root, rootIsZeroCost] = info.addRootedOperation(type, numOps);
  if (!root.has_value())
    return nullptr;
  builder.create<func::ReturnOp>(unknownLoc, *root);

  return module;
}

int main(int argc, char **argv) {

  // The IRDL file containing the dialects that we want to generate
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input program>"),
      llvm::cl::init("-"));

  // The IRDL file containing the dialects that we want to generate
  static llvm::cl::opt<std::string> inputIRDLFilename(
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

  static llvm::cl::opt<bool> printOpGeneric(
      "mlir-print-op-generic",
      llvm::cl::desc("Print the generic form of the operations"),
      llvm::cl::init(false));

  static llvm::cl::opt<bool> useInputOps(
      "use-input-ops",
      llvm::cl::desc("Use the input operations to enumerate programs"),
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
      "bit-vector-widths",
      llvm::cl::desc("In case the configuration is set to \"smt\", this is a "
                     "list of comma-separated bitwidths. If not specified, "
                     "this corresponds to no BitVector instructions."),
      llvm::cl::init(""));

  static llvm::cl::opt<bool> optimization(
      "optimize",
      llvm::cl::desc(
          "Only generate programs with a lower cost than the input program"),
      llvm::cl::init(false));

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR superoptimizer");

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

  // Parse the input program we want to optimize.
  auto optInputProgram = parseMLIRFile(ctx, inputFilename);
  if (!optInputProgram)
    return 1;
  auto &inputProgram = optInputProgram.value();

  // Get the unique function from the input program.
  SmallVector<func::FuncOp, 1> op =
      llvm::to_vector(inputProgram->getOps<func::FuncOp>());
  if (op.size() != 1) {
    llvm::errs() << "Expected exactly one function in the input program\n";
    return 1;
  }
  auto inputFunc = op[0];

  assert(inputFunc.getFunctionType().getNumResults() == 1 &&
         "Expected exactly one result in the input function");
  if (optimization) {
    int numInputOps = -2; // Do not count the function and the return op.
    inputFunc.walk([&numInputOps](Operation *op) {
      if (op->getName().getStringRef() != "arith.constant" &&
          op->getName().getStringRef() != "hw.constant")
        numInputOps += 1;
    });
    maxNumOps = std::min((int)maxNumOps, numInputOps);
  }

  // Try to parse the dialects.
  auto optDialects = parseMLIRFile(ctx, inputIRDLFilename);
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

  // Create the correct guide depending on the chosen strategy
  auto guide = tree_guide::EnumeratingGuide();

  auto createValueOutOfThinAir = [&ctx](GeneratorInfo &info,
                                        Type type) -> std::optional<Value> {
    OperationState state(UnknownLoc::get(&ctx), "synth.constant", {}, {type});
    auto op = info.builder.create(state);
    return op->getResult(0);
  };

  while (auto chooser = guide.makeChooser()) {
    auto module = createProgramFromInput(
        ctx, inputFunc, availableOps,
        getAvailableTypes(ctx, configuration, smtBvWidths),
        getAvailableAttributes(ctx, configuration), chooser.get(), maxNumOps,
        createValueOutOfThinAir, useInputOps, configuration);
    if (!module)
      continue;

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
    module->print(llvm::outs(), printingFlags);
    llvm::outs() << "// -----\n";
    llvm::outs().flush();

    if (pauseBetweenPrograms) {
      char c;
      std::cin >> c;
      if (c == 'q')
        break;
    }
  }
}
