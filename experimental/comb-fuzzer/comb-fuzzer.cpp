//===- tblgen-extract.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "guide.h"

#include "GeneratorInfo.h"
#include "Graph.h"

#include "mlir/InitAllDialects.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

using namespace mlir;

/// Create a value of the given type.
/// This may add a new argument to the function.
Value createValue(GeneratorInfo &info, int fuel, Type type) {
  if (fuel == 0)
    return getValue(info, type);
  assert(false && "Not implemented");
}

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

  GeneratorInfo info(chooser, {}, builder,
                     ctx.getOrLoadDialect<irdl::IRDLDialect>()->irdlContext);
  for (int i = 0; i < fuel; i++) {
    createValue(info, 0, IntegerType::get(&ctx, chooser->choose(7)));
  }

  builder.create<func::ReturnOp>(unknownLoc);
  return module;
}

int main(int argc, char **argv) {

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
  ctx.loadAllAvailableDialects();

  auto guide = tree_guide::BFSGuide(42);

  int n = 0;
  while (auto chooser = guide.makeChooser()) {
    auto module = createProgram(ctx, chooser.get(), 3);
    module->print(llvm::outs());
    llvm::errs() << "Printed " << n << "modules"
                 << "\n";
    n++;
  }

  llvm::errs() << n << " modules generated\n";

  return 0;
}