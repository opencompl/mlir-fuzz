//===- tblgen-extract.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "guide.h"

#include "Dyn/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace irdl;

/// Data structure to hold some information about the current program
/// being generated.
struct GeneratorInfo {
  /// The chooser, which will chose which path to take in the decision tree.
  tree_guide::Chooser *chooser;

  /// A builder set to the end of the function.
  OpBuilder builder;

  /// The set of values that are dominating the insertion point.
  /// We group the values by their type.
  /// We store values of the same type in a vector to iterate on them
  /// deterministically.
  /// Since we are iterating from top to bottom of the program, we do not
  /// need to remove elements from this set.
  llvm::DenseMap<Type, std::vector<Value>> dominatingValues;

  GeneratorInfo(tree_guide::Chooser *chooser, OpBuilder builder)
      : chooser(chooser), builder(builder) {}

  /// Add a value to the list of available values.
  void addDominatingValue(Value value) {
    dominatingValues[value.getType()].push_back(value);
  }
};

/// Get a value in the program.
/// This may add a new argument to the function.
Value getValue(GeneratorInfo &info, Type type) {
  auto builder = info.builder;
  auto &domValues = info.dominatingValues[type];

  // For now, we assume that we are only generating values of the same type.
  auto choice = info.chooser->choose(domValues.size() + 1);

  // If we chose a dominating value, return it
  if (choice < (long)domValues.size()) {
    return domValues[choice];
  }

  // Otherwise, add a new argument to the parent function.
  auto func = cast<func::FuncOp>(*builder.getInsertionBlock()->getParentOp());

  // We first chose an index where to add this argument.
  // Note that this is very costly when we are enumerating all programs of
  // a certain size.
  auto newArgIndex = info.chooser->choose(func.getNumArguments() + 1);

  func.insertArgument(newArgIndex, type, {},
                      UnknownLoc::get(builder.getContext()));
  auto arg = func.getArgument(newArgIndex);
  info.addDominatingValue(arg);
  return arg;
}

/// Add a random operation at the insertion point.
void addOperation(GeneratorInfo &info) {
  auto builder = info.builder;
  auto ctx = builder.getContext();
  std::vector<StringRef> availableOps = {"arith.addi", "arith.muli"};

  // Choose one of the binary operations.
  auto opName = availableOps[info.chooser->choose(availableOps.size())];

  // Choose the operands.
  auto lhs = getValue(info, builder.getIntegerType(32));
  auto rhs = getValue(info, builder.getIntegerType(32));

  // Create the operation.
  auto *op = builder.create(UnknownLoc::get(ctx), StringAttr::get(ctx, opName),
                            {lhs, rhs}, {builder.getIntegerType(32)});
  info.addDominatingValue(op->getResult(0));
}

/// Create a random program, given the decisions taken from chooser.
/// The program has at most `fuel` operations.
OwningOpRef<ModuleOp> createProgram(MLIRContext &ctx,
                                    tree_guide::Chooser *chooser, int fuel) {
  // Create an empty module.
  auto unknownLoc = UnknownLoc::get(&ctx);
  OwningOpRef<ModuleOp> module(ModuleOp::create(unknownLoc));

  // Create the builder, and set its insertion point in the module.
  OpBuilder builder(&ctx);
  auto &moduleBlock = module->getRegion().getBlocks().front();
  builder.setInsertionPoint(&moduleBlock, moduleBlock.begin());

  // Create an empty function, and set the insertion point in it.
  auto func = builder.create<func::FuncOp>(unknownLoc, "foo",
                                           FunctionType::get(&ctx, {}, {}));
  func.setPrivate();
  auto &funcBlock = func.getBody().emplaceBlock();
  builder.setInsertionPoint(&funcBlock, funcBlock.begin());

  // Create the generator info
  GeneratorInfo info(chooser, builder);

  // Select how many operations we want to generate, and generate them.
  auto numOps = chooser->choose(fuel + 1);
  for (long i = 0; i < numOps; i++) {
    addOperation(info);
  }

  builder.create<func::ReturnOp>(unknownLoc);
  return module;
}

/// Parse a file containing the dialects that we want to use.
llvm::Optional<OwningOpRef<ModuleOp>>
parseIRDLDialects(MLIRContext &ctx, StringRef inputFilename) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return llvm::None;
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

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR enumerator");

  MLIRContext ctx;

  // Register all dialects
  DialectRegistry registry;
  registerAllDialects(registry);
  ctx.appendDialectRegistry(registry);
  ctx.getOrLoadDialect<irdl::IRDLDialect>();
  ctx.loadAllAvailableDialects();

  llvm::errs() << "Parsing IRDL file" << '\n';

  // Try to parse the dialects.
  auto optDialects = parseIRDLDialects(ctx, inputFilename);
  if (!optDialects)
    return 1;

  // Get the dialects
  auto &dialects = optDialects.value();
  dialects->dump();

  auto guide = tree_guide::BFSGuide(42);
  while (auto chooser = guide.makeChooser()) {
    auto module = createProgram(ctx, chooser.get(), 2);
    module->dump();
  }
}
