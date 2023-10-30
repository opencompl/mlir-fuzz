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
  return {builder.getIntegerType(1),  builder.getIntegerType(32),
          builder.getIntegerType(64), builder.getIntegerType(127),
          builder.getIntegerType(17), builder.getIntegerType(3),
          builder.getIndexType()};
}

Value createIntegerValue(GeneratorInfo &info, IntegerType type) {
  auto ctx = info.builder.getContext();
  const std::vector<IntegerAttr> interestingValueList = {
      IntegerAttr::get(type, -1), IntegerAttr::get(type, 0),
      IntegerAttr::get(type, 1)};
  size_t choice = info.chooser->choose(interestingValueList.size() + 1);
  IntegerAttr value;
  if (choice == interestingValueList.size()) {
    value = IntegerAttr::get(type, info.chooser->chooseUnimportant());
  } else {
    value = interestingValueList[choice];
  }
  auto typedValue = value.cast<TypedAttr>();
  auto constant =
      info.builder.create<arith::ConstantOp>(UnknownLoc::get(ctx), typedValue);
  return constant.getResult();
}

std::optional<Value> createValueOutOfThinAir(GeneratorInfo &info, Type type) {
  if (auto intType = type.dyn_cast<IntegerType>())
    return createIntegerValue(info, intType);
  return {};
}

/// Add a random operation at the insertion point.
/// Return failure if no operations were added.
LogicalResult addOperation(GeneratorInfo &info, ArrayRef<Type> availableTypes) {
  auto builder = info.builder;
  auto ctx = builder.getContext();

  // Chose one operation between all available operations.
  auto availableOps = info.availableOps;
  auto op = availableOps[info.chooser->choose(availableOps.size())];

  // Get the IRDL verifier for this operation.
  auto [constraints, valueToIdx] = getOperationVerifier(op);
  ConstraintVerifier verifier(constraints);

  auto operandsOp = op.getOp<OperandsOp>();
  SmallVector<Value> operands = {};
  if (operandsOp) {
    for (Value operand : operandsOp->getArgs()) {
      auto satisfyingTypes = getSatisfyingTypes(*ctx, valueToIdx[operand],
                                                verifier, availableTypes);
      if (satisfyingTypes.size() == 0)
        return failure();
      auto type = satisfyingTypes[info.chooser->choose(satisfyingTypes.size())];

      // Set the operand variable in the verifier context, so other variables
      // are recursively set.
      auto verified =
          verifier.verify({}, TypeAttr::get(type), valueToIdx[operand]);
      assert(verified.succeeded());

      auto value = info.getValue(type);
      if (!value.has_value()) {
        return failure();
      }
      operands.push_back(*value);
    }
  }

  auto resultsOp = op.getOp<ResultsOp>();
  SmallVector<Type> resultTypes = {};
  if (resultsOp) {
    for (Value result : resultsOp->getArgs()) {
      auto satisfyingTypes = getSatisfyingTypes(*ctx, valueToIdx[result],
                                                verifier, availableTypes);
      if (satisfyingTypes.size() == 0)
        return failure();
      auto type = satisfyingTypes[info.chooser->choose(satisfyingTypes.size())];

      // Set the result variable in the verifier context, so other variables
      // are recursively set.
      auto verified =
          verifier.verify({}, TypeAttr::get(type), valueToIdx[result]);
      assert(verified.succeeded());

      resultTypes.push_back(type);
    }
  }

  StringRef dialectName = op.getParentOp().getName();
  StringRef opSuffix = op.getNameAttr().getValue();
  StringAttr opName = StringAttr::get(ctx, dialectName + "." + opSuffix);

  // Create the operation.
  auto *operation =
      builder.create(UnknownLoc::get(ctx), opName, operands, resultTypes);
  for (auto result : operation->getResults()) {
    info.addDominatingValue(result);
  }

  return success();
}

/// Create a random program, given the decisions taken from chooser.
/// The program has at most `fuel` operations.
OwningOpRef<ModuleOp> createProgram(MLIRContext &ctx,
                                    ArrayRef<OperationOp> availableOps,
                                    tree_guide::Chooser *chooser, int numOps,
                                    int numArgs) {
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
  auto &funcBlock = func.getBody().emplaceBlock();
  builder.setInsertionPoint(&funcBlock, funcBlock.begin());

  // Create the generator info
  GeneratorInfo info(chooser, availableOps, builder);

  auto availableTypes = getAvailableTypes(ctx);
  // Add function arguments
  for (int i = 0; i < numArgs; i++) {
    auto type = availableTypes[chooser->choose(availableTypes.size())];
    info.addFunctionArgument(type);
  }

  // Select how many operations we want to generate, and generate them.
  for (long i = 0; i < numOps; i++) {
    if (addOperation(info, getAvailableTypes(ctx)).failed())
      return nullptr;
  }

  std::vector<Value> values;
  for (auto v : info.dominatingValues)
    values.insert(values.end(), v.second.begin(), v.second.end());

  if (values.empty())
    return module;

  auto value = values[chooser->choose(values.size())];
  builder.create<func::ReturnOp>(unknownLoc, value);
  func.insertResult(0, value.getType(), {});
  return module;
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

  auto guide = tree_guide::DefaultGuide();
  // auto guide = tree_guide::BFSGuide(42);
  while (auto chooser = guide.makeChooser()) {
    auto module = createProgram(ctx, availableOps, chooser.get(), 10, 3);
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
