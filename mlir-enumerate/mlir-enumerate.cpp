//===- tblgen-extract.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "GeneratorInfo.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IRDLContext.h"
#include "mlir/Dialect/IRDL/IRDLVerifiers.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
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

/// Get the types that the fuzzer supports.
std::vector<Type> getAvailableTypes(MLIRContext &ctx) {
  Builder builder(&ctx);
  return {builder.getIntegerType(1),  builder.getIntegerType(32),
          builder.getIntegerType(64), builder.getIntegerType(17),
          builder.getIntegerType(3),  builder.getIndexType(),
          builder.getF32Type(),       builder.getF64Type()};
}

/// Get the types that the constraint can support, given a constraint context.
std::vector<Type> getSatisfyingTypes(MLIRContext &ctx, int constraint,
                                     irdl::ConstraintVerifier &context) {
  std::vector<Type> availableType = getAvailableTypes(ctx);

  std::vector<Type> satisfyingTypes;
  for (auto type : availableType) {
    irdl::ConstraintVerifier context_copy = context;
    if (context_copy.verify({}, TypeAttr::get(type), constraint).succeeded()) {
      satisfyingTypes.push_back(type);
    }
  }
  return satisfyingTypes;
}

/// Add a random operation at the insertion point.
/// Return failure if no operations were added.
LogicalResult addOperation(GeneratorInfo &info) {
  auto builder = info.builder;
  auto ctx = builder.getContext();

  DenseMap<TypeOp, std::unique_ptr<DynamicTypeDefinition>> types;
  DenseMap<AttributeOp, std::unique_ptr<DynamicAttrDefinition>> attrs;

  // Chose one operation between all available operations.
  auto availableOps = info.availableOps;
  auto op = availableOps[info.chooser->choose(availableOps.size())];

  // Resolve SSA values to verifier constraint slots
  SmallVector<Value> constrToValue;
  DenseMap<Value, int> valueToIdx;
  for (Operation &op : op->getRegion(0).getOps()) {
    if (isa<VerifyConstraintInterface>(op)) {
      assert(op.getNumResults() == 1);
      valueToIdx[op.getResult(0)] = constrToValue.size();
      constrToValue.push_back(op.getResult(0));
    }
  }

  // Build the verifiers for each constraint slot
  SmallVector<std::unique_ptr<Constraint>> constraints;
  DenseMap<Value, Constraint *> valueToConstraint;
  for (Value v : constrToValue) {
    VerifyConstraintInterface op =
        cast<VerifyConstraintInterface>(v.getDefiningOp());
    std::unique_ptr<Constraint> verifier =
        op.getVerifier(constrToValue, types, attrs);
    assert(verifier && "Constraint verifier couldn't be generated");
    valueToConstraint[v] = verifier.get();
    constraints.push_back(std::move(verifier));
  }

  // The verifier, that will check that the operands/results satisfy the
  // invariants.
  ConstraintVerifier verifier(constraints);

  auto operandsOp = op.getOp<OperandsOp>();
  SmallVector<Value> operands = {};
  if (operandsOp) {
    for (Value operand : operandsOp->getArgs()) {
      auto operandConstraint = valueToConstraint[operand];
      auto satisfyingTypes =
          getSatisfyingTypes(*ctx, valueToIdx[operand], verifier);
      if (satisfyingTypes.size() == 0)
        return failure();
      auto type = satisfyingTypes[info.chooser->choose(satisfyingTypes.size())];

      // Set the operand variable in the verifier context, so other variables
      // are recursively set.
      auto verified =
          verifier.verify({}, TypeAttr::get(type), valueToIdx[operand]);
      assert(verified.succeeded());

      operands.push_back(getValue(info, type));
    }
  }

  auto resultsOp = op.getOp<ResultsOp>();
  SmallVector<Type> resultTypes = {};
  if (resultsOp) {
    for (Value result : resultsOp->getArgs()) {
      auto resultConstraint = valueToConstraint[result];
      auto satisfyingTypes =
          getSatisfyingTypes(*ctx, valueToIdx[result], verifier);
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

  // Create the operation.
  auto *operation = builder.create(UnknownLoc::get(ctx), op.getNameAttr(),
                                   operands, resultTypes);
  for (auto result : operation->getResults()) {
    info.addDominatingValue(result);
  }

  return success();
}

/// Create a random program, given the decisions taken from chooser.
/// The program has at most `fuel` operations.
OwningOpRef<ModuleOp> createProgram(MLIRContext &ctx,
                                    ArrayRef<OperationOp> availableOps,
                                    IRDLContext &irdlCtx,
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
  GeneratorInfo info(chooser, availableOps, builder, irdlCtx);

  // Select how many operations we want to generate, and generate them.
  auto numOps = chooser->choose(fuel + 1);
  for (long i = 0; i < numOps; i++) {
    if (addOperation(info).failed())
      return nullptr;
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

class IntegerTypeWrapper : public CppTypeWrapper<IntegerType> {
  StringRef getName() override { return "builtin.integer_type"; }

  /// Instanciates the type from parameters.
  Type instantiateType(llvm::function_ref<InFlightDiagnostic()> emitError,
                       llvm::ArrayRef<Attribute> parameters) override {
    if (parameters.size() != 1) {
      emitError() << "expected 1 parameter, got " << parameters.size();
      return {};
    }

    auto widthAttr = parameters[0].dyn_cast<IntegerAttr>();
    if (!widthAttr) {
      emitError() << "expected integer attribute parameter, got "
                  << parameters[0];
      return {};
    }

    auto *ctx = widthAttr.getContext();
    Builder builder(ctx);
    return builder.getIntegerType(widthAttr.getInt());
  }

  size_t getParameterAmount() override { return 1; }

  llvm::SmallVector<mlir::Attribute>
  getTypeParameters(IntegerType type) override {
    auto width = type.getWidth();
    auto *context = type.getContext();
    Builder builder(context);
    return {builder.getIndexAttr(width)};
  }
};

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
  auto *irdlDialect = ctx.getOrLoadDialect<irdl::IRDLDialect>();
  ctx.loadAllAvailableDialects();

  irdlDialect->irdlContext.addTypeWrapper(
      std::make_unique<IntegerTypeWrapper>());
  auto &irdlContext = irdlDialect->irdlContext;

  // Try to parse the dialects.
  auto optDialects = parseIRDLDialects(ctx, inputFilename);
  if (!optDialects)
    return 1;

  // Get the dialects.
  auto &dialects = optDialects.value();
  dialects->dump();

  // Get the list of operations we support.
  std::vector<OperationOp> availableOps = {};
  dialects->walk([&availableOps, &irdlContext](OperationOp op) {
    availableOps.push_back(op);
  });

  size_t programCounter = 0;
  size_t correctProgramCounter = 0;

  auto guide = tree_guide::BFSGuide(42);
  while (auto chooser = guide.makeChooser()) {
    auto module =
        createProgram(ctx, availableOps, irdlContext, chooser.get(), 2);
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
    llvm::errs() << "Generated " << programCounter << " programs, "
                 << (((float)correctProgramCounter / (float)programCounter) *
                     100.0f)
                 << "% verifying \n\n\n";

    if (outputFolder == "-") {
      module->dump();
      continue;
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
