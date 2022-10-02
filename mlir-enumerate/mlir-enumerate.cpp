//===- tblgen-extract.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "guide.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IR/IRDLAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace irdl;

/// Data structure to hold some information about the current program
/// being generated.
struct GeneratorInfo {
  /// The chooser, which will chose which path to take in the decision tree.
  tree_guide::Chooser *chooser;

  /// All available ops that can be used by the fuzzer.
  ArrayRef<OperationOp> availableOps;

  /// A builder set to the end of the function.
  OpBuilder builder;

  /// Context for the runtime registration of IRDL dialect definitions.
  IRDLContext &irdlContext;

  /// The set of values that are dominating the insertion point.
  /// We group the values by their type.
  /// We store values of the same type in a vector to iterate on them
  /// deterministically.
  /// Since we are iterating from top to bottom of the program, we do not
  /// need to remove elements from this set.
  llvm::DenseMap<Type, std::vector<Value>> dominatingValues;

  GeneratorInfo(tree_guide::Chooser *chooser,
                ArrayRef<OperationOp> availableOps, OpBuilder builder,
                IRDLContext &irdlContext)
      : chooser(chooser), availableOps(availableOps), builder(builder),
        irdlContext(irdlContext) {}

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

/// Get the types that the fuzzer supports.
std::vector<Type> getAvailableTypes(MLIRContext &ctx) {
  Builder builder(&ctx);
  return {builder.getIntegerType(1),  builder.getIntegerType(32),
          builder.getIntegerType(64), builder.getIntegerType(17),
          builder.getIntegerType(3),  builder.getIndexType(),
          builder.getF32Type(),       builder.getF64Type()};
}

/// Get the types that the constraint can support.
std::vector<Type> getSatisfyingTypes(
    MLIRContext &ctx, irdl::TypeConstraint *constraint,
    ArrayRef<std::unique_ptr<irdl::TypeConstraint>> varConstraints,
    MutableArrayRef<Type> vars) {
  std::vector<Type> availableType = getAvailableTypes(ctx);

  std::vector<Type> satisfyingTypes;
  for (auto type : availableType) {
    if (constraint->verifyType({}, type, varConstraints, vars).succeeded()) {
      satisfyingTypes.push_back(type);
    }
  }
  return satisfyingTypes;
}

/// Get the types that the constraint can support.
std::vector<Type> getSatisfyingTypes(
    IRDLContext &irdlCtx, TypeConstraintAttrInterface constraintAttr,
    ArrayRef<std::unique_ptr<irdl::TypeConstraint>> varConstraints,
    MutableArrayRef<Type> vars,
    const SmallVector<
        std::pair<StringRef, std::unique_ptr<irdl::TypeConstraint>>>
        &namedVars) {
  auto *ctx = constraintAttr.getContext();
  Builder builder(ctx);

  auto constr = constraintAttr.getTypeConstraint(irdlCtx, namedVars);
  return getSatisfyingTypes(*ctx, constr.get(), varConstraints, vars);
}

/// Get the types that the constraint can support.
std::vector<Type> getSatisfyingTypes(
    IRDLContext &irdlCtx, NamedTypeConstraintAttr constraint,
    ArrayRef<std::unique_ptr<irdl::TypeConstraint>> varConstraints,
    MutableArrayRef<Type> vars,
    const SmallVector<
        std::pair<StringRef, std::unique_ptr<irdl::TypeConstraint>>>
        &namedVars) {
  return getSatisfyingTypes(
      irdlCtx, constraint.getConstraint().cast<TypeConstraintAttrInterface>(),
      varConstraints, vars, namedVars);
}

/// Get the types that the constraint can support.
std::vector<Type> getSatisfyingTypes(
    IRDLContext &irdlCtx, Attribute constraint,
    ArrayRef<std::unique_ptr<irdl::TypeConstraint>> varConstraints,
    MutableArrayRef<Type> vars,
    const SmallVector<
        std::pair<StringRef, std::unique_ptr<irdl::TypeConstraint>>>
        &namedVars) {
  if (auto constr = constraint.dyn_cast<NamedTypeConstraintAttr>()) {
    return getSatisfyingTypes(irdlCtx, constr, varConstraints, vars, namedVars);
  } else if (auto constr = constraint.dyn_cast<TypeConstraintAttrInterface>()) {
    return getSatisfyingTypes(irdlCtx, constr, varConstraints, vars, namedVars);
  }
  assert(false && "Unknown attribute given to getSatisfyingTypes");
}

/// Add a random operation at the insertion point.
/// Return failure if no operations were added.
LogicalResult addOperation(GeneratorInfo &info) {
  auto builder = info.builder;
  auto ctx = builder.getContext();

  // Chose one operation between all available operations.
  auto availableOps = info.availableOps;
  auto op = availableOps[info.chooser->choose(availableOps.size())];

  // The constraint variables, and their assignment.
  SmallVector<std::pair<StringRef, std::unique_ptr<irdl::TypeConstraint>>>
      namedConstraintVars = {};
  SmallVector<std::unique_ptr<irdl::TypeConstraint>> varConstraints;
  SmallVector<Type> vars;

  // For each constraint variable, we assign a type.
  auto constraintOp = op.getOp<ConstraintVarsOp>();
  if (constraintOp) {
    for (auto namedConstraintAttr : constraintOp->getParams()) {
      auto namedConstraint =
          namedConstraintAttr.cast<NamedTypeConstraintAttr>();
      auto constraint =
          namedConstraint.getConstraint()
              .cast<TypeConstraintAttrInterface>()
              .getTypeConstraint(info.irdlContext, namedConstraintVars);
      // TODO(fehr) Currently a hack, will be fixed later once I update
      // the IRDL API.
      auto constraint2 =
          namedConstraint.getConstraint()
              .cast<TypeConstraintAttrInterface>()
              .getTypeConstraint(info.irdlContext, namedConstraintVars);

      auto satisfyingTypes =
          getSatisfyingTypes(*ctx, constraint.get(), varConstraints, vars);
      if (satisfyingTypes.size() == 0)
        return failure();
      auto type = satisfyingTypes[info.chooser->choose(satisfyingTypes.size())];

      namedConstraintVars.emplace_back(namedConstraint.getName(),
                                       std::move(constraint));
      varConstraints.emplace_back(std::move(constraint2));
      vars.push_back(type);
    }
  }

  auto operandDefs = op.getOp<OperandsOp>();
  SmallVector<Value> operands = {};
  if (operandDefs) {
    for (auto operandAttr : operandDefs->getParams()) {
      auto operandConstr = operandAttr.cast<NamedTypeConstraintAttr>();
      auto satisfyingTypes =
          getSatisfyingTypes(info.irdlContext, operandConstr, varConstraints,
                             vars, namedConstraintVars);
      if (satisfyingTypes.size() == 0)
        return failure();
      auto type = satisfyingTypes[info.chooser->choose(satisfyingTypes.size())];
      operands.push_back(getValue(info, type));
    }
  }

  auto resultDefs = op.getOp<ResultsOp>();
  SmallVector<Type> resultTypes = {};
  if (resultDefs) {
    for (auto resultAttr : resultDefs->getParams()) {
      auto resultConstr = resultAttr.cast<NamedTypeConstraintAttr>();
      auto satisfyingTypes =
          getSatisfyingTypes(info.irdlContext, resultConstr, varConstraints,
                             vars, namedConstraintVars);
      if (satisfyingTypes.size() == 0)
        return failure();
      auto type = satisfyingTypes[info.chooser->choose(satisfyingTypes.size())];
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

class IntegerTypeWrapper : public CppTypeWrapper<IntegerType> {
  StringRef getName() override { return "builtin.integer_type"; }

  /// Instanciates the type from parameters.
  Type instantiate(llvm::function_ref<InFlightDiagnostic()> emitError,
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

  llvm::SmallVector<mlir::Attribute> getParameters(IntegerType type) override {
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

  irdlDialect->addTypeWrapper(std::make_unique<IntegerTypeWrapper>());
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
