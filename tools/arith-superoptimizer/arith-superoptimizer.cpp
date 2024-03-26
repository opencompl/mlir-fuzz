//===- arith-superoptimizer.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <filesystem>
#include <sys/wait.h>
#include <unistd.h>

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

bool executeAndSaveModule(mlir::ModuleOp module, StringRef outputDirectory) {
  auto arithModule = module.clone();

  // Convert the module to LLVM IR
  if (convertModuleToLLVM(module).failed()) {
    llvm::errs() << "Failed to convert the module to LLVM IR\n";
    module->print(llvm::errs());
    return false;
  }

  // Create an execution engine
  auto engine = ExecutionEngine::create(module);
  if (auto error = engine.takeError()) {
    llvm::errs() << "Failed to create an execution engine for \n";
    module->print(llvm::errs());
    llvm::errs() << error;

    return false;
  }

  auto func = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("main");
  auto seed =
      func->getAttrOfType<mlir::IntegerAttr>("seed").getValue().getSExtValue();
  int numArguments = func.getNumArguments();
  assert(numArguments == 1);

  std::vector<int32_t> interestingValues = {0, 1, -1, 2, 7, -12, 64, 1235987};

  pid_t c_pid = fork(); // fork a child process

  // The child process run the program, and save it
  if (c_pid == 0) {
    std::string hashStr;
    for (int32_t input : interestingValues) {
      std::vector<void *> args;
      args.push_back(&input);
      int32_t result;
      args.push_back(&result);

      auto invokationError = engine->get()->invokePacked("main", args);
      if (invokationError) {
        llvm::errs() << "Failed to invoke the main function for ";
        module->print(llvm::errs());
        exit(1);
      }
      // use the result to compute the hashStr
      hashStr = hashStr + "_" + std::to_string(result);
    }
    if (outputDirectory.empty())
      exit(0);

    // Printing flags
    OpPrintingFlags printingFlags;
    printingFlags.printGenericOpForm(true);

    // create a new directory with the hash if needed
    std::string directory = (outputDirectory + "/hash" + hashStr).str();
    std::filesystem::create_directories(directory);
    std::string seedStr = std::to_string(seed);
    std::string filename = directory + "/module" + seedStr + ".mlir";
    std::error_code error;
    llvm::raw_fd_ostream file(filename, error);
    if (error) {
      llvm::errs() << "Failed to open the file " << filename
                   << " for writing\n";
      exit(1);
    }
    arithModule->print(file, printingFlags);
    file.close();
    exit(0);
  } else {
    // The current process waits for the child process to finish,
    // and just checks that the program did not crash.
    int status;
    waitpid(c_pid, &status, 0);
    return status == 0;
  }
}

int main(int argc, char **argv) {

  // The IRDL file containing the dialects that we want to generate
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<IRDL file>"), llvm::cl::init("-"));
  static llvm::cl::opt<std::string> outputDirectory(
      "output-directory", llvm::cl::desc("Output directory"),
      llvm::cl::value_desc("directory"), llvm::cl::init(""));

  // Number of non-constant operations to be printed.
  static llvm::cl::opt<int> maxNumOps(
      "max-num-ops",
      llvm::cl::desc("Maximum number of non-constant operations"),
      llvm::cl::init(2));

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
  size_t executedProgramCounter = 0;

  std::random_device rd;
  std::uniform_int_distribution<int> dist(0, 1 << 30);

  auto guide = tree_guide::BFSGuide();
  while (auto chooser = guide.makeChooser()) {
    auto module = createProgram(ctx, availableOps, availableTypes(ctx),
                                getAvailableAttributes(ctx), chooser.get(),
                                maxNumOps, 1, correctProgramCounter);
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

    if (executeAndSaveModule(module.get(), outputDirectory))
      executedProgramCounter += 1;

    if (correctProgramCounter % 10 == 0) {
      llvm::outs() << "Generated " << programCounter << " programs, "
                   << correctProgramCounter << " of which verify, "
                   << executedProgramCounter
                   << " of which were executed without any bugs.\n";
    }
  }

  llvm::outs() << "Generated " << programCounter << " programs, "
               << correctProgramCounter << " of which verify, "
               << executedProgramCounter
               << " of which were executed without any bugs.\n";
}
