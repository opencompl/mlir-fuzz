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
#include "Dyn/Dialect/IRDL/IR/IRDLAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  return {builder.getIntegerType(32)};
}

/// Get the types that the constraint can support.
std::vector<Type>
getSatisfyingTypes(IRDLContext &irdlCtx,
                   TypeConstraintAttrInterface constraintAttr) {
  auto *ctx = constraintAttr.getContext();
  Builder builder(ctx);

  std::vector<Type> availableType = getAvailableTypes(*ctx);
  auto constr = constraintAttr.getTypeConstraint(irdlCtx, {});

  std::vector<Type> satisfyingTypes;
  for (auto type : availableType) {
    if (constr->verifyType({}, type, {}, {}).succeeded()) {
      satisfyingTypes.push_back(type);
    }
  }
  return satisfyingTypes;
}

/// Get the types that the constraint can support.
std::vector<Type> getSatisfyingTypes(IRDLContext &irdlCtx,
                                     NamedTypeConstraintAttr constraint) {
  return getSatisfyingTypes(
      irdlCtx, constraint.getConstraint().cast<TypeConstraintAttrInterface>());
}

/// Get the types that the constraint can support.
std::vector<Type> getSatisfyingTypes(IRDLContext &irdlCtx,
                                     Attribute constraint) {
  if (auto constr = constraint.dyn_cast<NamedTypeConstraintAttr>()) {
    return getSatisfyingTypes(irdlCtx, constr);
  } else if (auto constr = constraint.dyn_cast<TypeConstraintAttrInterface>()) {
    return getSatisfyingTypes(irdlCtx, constr);
  }
  assert(false && "Unknown attribute given to getSatisfyingTypes");
}

/// Add a random operation at the insertion point.
void addOperation(GeneratorInfo &info) {
  auto builder = info.builder;
  auto ctx = builder.getContext();

  auto availableOps = info.availableOps;

  auto op = availableOps[info.chooser->choose(availableOps.size())];

  auto operandDefs = op.getOp<OperandsOp>();
  SmallVector<Value> operands = {};
  if (operandDefs) {
    for (auto operandAttr : operandDefs->params()) {
      auto operandConstr = operandAttr.cast<NamedTypeConstraintAttr>();
      auto satisfyingTypes =
          getSatisfyingTypes(info.irdlContext, operandConstr);
      auto type = satisfyingTypes[info.chooser->choose(satisfyingTypes.size())];
      operands.push_back(getValue(info, type));
    }
  }

  auto resultDefs = op.getOp<OperandsOp>();
  SmallVector<Type> resultTypes = {};
  if (resultDefs) {
    for (auto resultAttr : resultDefs->params()) {
      auto resultConstr = resultAttr.cast<NamedTypeConstraintAttr>();
      auto satisfyingTypes = getSatisfyingTypes(info.irdlContext, resultConstr);
      auto type = satisfyingTypes[info.chooser->choose(satisfyingTypes.size())];
      resultTypes.push_back(type);
    }
  }

  // Create the operation.
  auto *operation = builder.create(UnknownLoc::get(ctx), op.nameAttr(),
                                   operands, resultTypes);
  for (auto result : operation->getResults()) {
    info.addDominatingValue(result);
  }
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

/// Check that we know types that can satisfy the operation constraints.
/// In an ideal implementation, we should be able to satisfy all of these
/// operation constraints.
bool canSatisfyOpConstrs(IRDLContext &irdlCtx, OperationOp op) {
  // Check if we can satisfy all the operand constraints.
  auto operandDefs = op.getOp<OperandsOp>();
  if (operandDefs) {
    for (auto operandAttr : operandDefs->params()) {
      auto operandConstr = operandAttr.cast<NamedTypeConstraintAttr>();
      if (getSatisfyingTypes(irdlCtx, operandConstr).empty()) {
        return false;
      }
    }
  }

  // Check if we can satisfy all the result constraints.
  auto resultDefs = op.getOp<ResultsOp>();
  if (resultDefs) {
    for (auto resultAttr : resultDefs->params()) {
      auto resultConstr = resultAttr.cast<NamedTypeConstraintAttr>();
      if (getSatisfyingTypes(irdlCtx, resultConstr).empty()) {
        return false;
      }
    }
  }

  return true;
}
class IntegerTypeWrapper : public ConcreteTypeWrapper<IntegerType> {
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

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR enumerator");

  MLIRContext ctx;

  // Register all dialects
  DialectRegistry registry;
  registerAllDialects(registry);
  ctx.appendDialectRegistry(registry);
  ctx.getOrLoadDialect<irdl::IRDLDialect>();
  ctx.loadAllAvailableDialects();

  IRDLContext irdlContext;
  irdlContext.addTypeWrapper(std::make_unique<IntegerTypeWrapper>());

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
    if (canSatisfyOpConstrs(irdlContext, op))
      availableOps.push_back(op);
  });

  auto guide = tree_guide::BFSGuide(42);
  while (auto chooser = guide.makeChooser()) {
    auto module =
        createProgram(ctx, availableOps, irdlContext, chooser.get(), 2);
    module->dump();
    (void)verify(*module, true);
    llvm::errs() << "\n";
  }
}
