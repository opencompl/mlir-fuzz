//===- arith-superoptimizer.cpp ---------------------------------*- C++ -*-===//
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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IRDLVerifiers.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"

using namespace mlir;
using namespace irdl;

LogicalResult convertModuleToLLVM(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(mlir::arith::createArithExpandOpsPass());
  pm.addPass(createArithToLLVMConversionPass());
  return pm.run(module);
}

std::vector<Type> availableTypes(MLIRContext &ctx) {
  Builder builder(&ctx);
  return {builder.getIntegerType(32)};
}

int main(int argc, char **argv) {

  // The IRDL file containing the dialects that we want to generate
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<IRDL file>"), llvm::cl::init("-"));

  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR enumerator");

  MLIRContext ctx;
  ctx.allowUnregisteredDialects();

  // Register all dialects
  DialectRegistry registry;
  registerAllDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
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
    auto module = createProgram(ctx, availableOps, availableTypes(ctx),
                                getAvailableAttributes(ctx), chooser.get(), 2,
                                0, correctProgramCounter);
    if (!module)
      continue;

    auto func = module->lookupSymbol<func::FuncOp>("main");
    assert(func && "main function not found");
    if (func.getNumArguments() != 1)
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

    if (convertModuleToLLVM(module.get()).failed()) {
      llvm::errs() << "Failed to convert the module to LLVM IR\n";
      module->print(llvm::errs());
      continue;
    }
    llvm::outs() << "Converted module to LLVM IR\n";
    module->print(llvm::outs());

    auto engine = ExecutionEngine::create(module.get());
    if (auto error = engine.takeError()) {
      llvm::errs() << "Failed to create an execution engine for \n";
      module->print(llvm::errs());
      llvm::errs() << error;

      continue;
    }

    llvm::errs() << "Running the main function for ";

    int32_t input = 42;
    int32_t result = 0;
    std::vector<void *> args;
    args.push_back(&input);
    args.push_back(&result);
    auto invokationError = engine->get()->invokePacked("main", args);
    if (invokationError) {
      llvm::errs() << "Failed to invoke the main function for ";
      module->print(llvm::errs());
      continue;
    }

    llvm::errs() << "Result: " << result << "\n";
  }
}
