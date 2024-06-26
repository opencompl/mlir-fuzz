//===- GeneratorInfo.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GeneratorInfo.h"
#include "IRDLUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::irdl;

GeneratorInfo::GeneratorInfo(
    tree_guide::Chooser *chooser, mlir::OpBuilder builder,
    mlir::ArrayRef<mlir::irdl::OperationOp> availableOps,
    mlir::ArrayRef<mlir::Type> availableTypes,
    mlir::ArrayRef<mlir::Attribute> availableAttributes, int maxNumArgs,
    GeneratorInfo::CreateValueOutOfThinAirFn createValueOutOfThinAirFn)
    : chooser(chooser), builder(builder), availableOps(availableOps),
      availableTypes(availableTypes), availableAttributes(availableAttributes),
      maxNumArgs(maxNumArgs),
      createValueOutOfThinAir(createValueOutOfThinAirFn) {
  if (!createValueOutOfThinAir) {
    createValueOutOfThinAir = [](GeneratorInfo &info,
                                 Type type) -> std::optional<Value> {
      auto func = llvm::cast<mlir::func::FuncOp>(
          *info.builder.getInsertionBlock()->getParentOp());
      if (func.getNumArguments() < (unsigned int)info.maxNumArgs &&
          info.chooser->choose(2) == 0)
        return info.addFunctionArgument(type);

      if (auto intType = type.dyn_cast<IntegerType>())
        return info.createIntegerValue(intType);
      return {};
    };
  }
}

/// Create a constant of the given integer type.
Value GeneratorInfo::createIntegerValue(IntegerType type) {
  auto ctx = builder.getContext();
  const std::vector<IntegerAttr> interestingValueList = {
      IntegerAttr::get(type, -1), IntegerAttr::get(type, 0),
      IntegerAttr::get(type, 1)};
  // TODO: Add options to generate these.
  size_t choice = chooser->choose(interestingValueList.size());
  IntegerAttr value;
  if (choice == interestingValueList.size()) {
    value = IntegerAttr::get(type, chooser->chooseUnimportant());
  } else {
    value = interestingValueList[choice];
  }
  auto typedValue = value.cast<TypedAttr>();
  auto constant =
      builder.create<arith::ConstantOp>(UnknownLoc::get(ctx), typedValue);
  return constant.getResult();
}

/// Return the list of operations that can have a particular result type as
/// result.
/// Returns as well the indices of the results that can have this result type.
std::vector<std::pair<OperationOp, std::vector<int>>>
GeneratorInfo::getOperationsWithResultType(Type resultType) {
  static DenseMap<Type, std::vector<std::pair<OperationOp, std::vector<int>>>>
      memoization;

  if (memoization.find(resultType) != memoization.end())
    return memoization[resultType];

  auto ctx = builder.getContext();

  // Choose one operation that can support the resulting type.
  // Also get the indices of result definitions that are satisfied by this
  // type.
  std::vector<std::pair<OperationOp, std::vector<int>>> res;

  for (auto op : availableOps) {
    std::vector<int> satisfiableResults;
    for (auto [idx, resultDef] : llvm::enumerate(getResultsConstraints(op))) {
      if (!getSatisfyingTypes(*ctx, resultDef, op, {resultType}).empty())
        satisfiableResults.push_back(idx);
    }
    if (satisfiableResults.empty())
      continue;
    res.push_back({op, satisfiableResults});
  }

  memoization[resultType] = res;

  return res;
}

static std::optional<Value> getZeroCostValue(GeneratorInfo &info, Type type) {
  auto &domValues = info.dominatingValues[type];
  bool canUseDominatedValue = domValues.size();

  if (canUseDominatedValue && info.chooser->choose(2) == 0) {
    auto value = info.getValue(type);
    assert(value && "Error in generator logic");
    return *value;
  }

  return info.createValueOutOfThinAir(info, type);
}

/// Add an operation with a given result type.
/// Return the result that has has the requested type.
/// This function will also create a number of operations less than `fuel`
/// operations.
std::optional<Value> GeneratorInfo::addRootedOperation(Type resultType,
                                                       int fuel) {
  auto ctx = builder.getContext();

  // When we don't have fuel anymore, we either use a dominated value,
  // or we create a value out of thin air, which may include adding
  // a new function argument.
  if (fuel == 0)
    return getZeroCostValue(*this, resultType);

  // Cost of the current operation being created.
  fuel -= 1;

  auto operations = getOperationsWithResultType(resultType);
  if (operations.empty())
    return getZeroCostValue(*this, resultType);

  auto [op, possibleResults] = operations[chooser->choose(operations.size())];
  size_t resultIdx = possibleResults[chooser->choose(possibleResults.size())];

  // Create a new verifier that will keep track of the values we have already
  // assigned.
  auto [constraints, valueToIdx] = getOperationVerifier(op);
  auto verifier = ConstraintVerifier(constraints);

  // Add to the constraint verifier our set result.
  auto succeeded =
      verifier.verify({}, TypeAttr::get(resultType),
                      valueToIdx[getResultsConstraints(op)[resultIdx]]);
  assert(succeeded.succeeded());

  std::vector<Value> operands;
  for (auto operand : getOperandsConstraints(op)) {
    auto satisfyingTypes =
        getSatisfyingTypes(*ctx, valueToIdx[operand], verifier, availableTypes);
    if (satisfyingTypes.empty())
      return {};

    auto type = satisfyingTypes[chooser->choose(satisfyingTypes.size())];
    auto succeeded =
        verifier.verify({}, TypeAttr::get(type), valueToIdx[operand]);
    assert(succeeded.succeeded());

    int operandFuel = chooser->choose(fuel + 1);
    fuel -= operandFuel;

    auto operandValue = addRootedOperation(type, operandFuel);
    if (!operandValue.has_value())
      return {};
    operands.push_back(*operandValue);
  }

  std::vector<Type> resultTypes;
  for (auto [idx, result] : llvm::enumerate(getResultsConstraints(op))) {
    if (resultIdx == idx) {
      resultTypes.push_back(resultType);
      continue;
    }

    auto satisfyingTypes =
        getSatisfyingTypes(*ctx, valueToIdx[result], verifier, availableTypes);
    if (satisfyingTypes.size() == 0)
      return {};

    auto type = satisfyingTypes[chooser->choose(satisfyingTypes.size())];
    auto succeeded =
        verifier.verify({}, TypeAttr::get(type), valueToIdx[result]);
    assert(succeeded.succeeded());
    resultTypes.push_back(type);
  }

  std::vector<NamedAttribute> attributes;
  for (auto [name, constraint] : getAttributesConstraints(op)) {
    auto satisfyingAttrs = getSatisfyingAttrs(*ctx, valueToIdx[constraint],
                                              verifier, availableAttributes);
    if (satisfyingAttrs.size() == 0)
      return {};

    auto attr = satisfyingAttrs[chooser->choose(satisfyingAttrs.size())];
    auto succeeded = verifier.verify({}, attr, valueToIdx[constraint]);
    assert(succeeded.succeeded());
    attributes.emplace_back(StringAttr::get(ctx, name), attr);
  }

  StringRef dialectName = op.getParentOp().getName();
  StringRef opSuffix = op.getNameAttr().getValue();
  StringAttr opName = StringAttr::get(ctx, dialectName + "." + opSuffix);

  // Create the operation.
  auto *operation = builder.create(UnknownLoc::get(ctx), opName, operands,
                                   resultTypes, attributes);
  for (auto result : operation->getResults()) {
    addDominatingValue(result);
  }

  return operation->getResult(resultIdx);
}

/// Create a random program, given the decisions taken from chooser.
/// The program has at most `fuel` operations.
OwningOpRef<ModuleOp> createProgram(
    MLIRContext &ctx, ArrayRef<OperationOp> availableOps,
    ArrayRef<Type> availableTypes, ArrayRef<Attribute> availableAttributes,
    tree_guide::Chooser *chooser, int numOps, int numArgs, int seed,
    GeneratorInfo::CreateValueOutOfThinAirFn createValueOutOfThinAir) {
  // Create an empty module.
  auto unknownLoc = UnknownLoc::get(&ctx);
  OwningOpRef<ModuleOp> module(ModuleOp::create(unknownLoc));

  // Create the builder, and set its insertion point in the module.
  OpBuilder builder(&ctx);
  auto &moduleBlock = module->getRegion().getBlocks().front();
  builder.setInsertionPoint(&moduleBlock, moduleBlock.begin());

  // Create an empty function, and set the insertion point in it.
  auto func = builder.create<func::FuncOp>(unknownLoc, "main",
                                           FunctionType::get(&ctx, {}, {}));
  func->setAttr("seed", IntegerAttr::get(IndexType::get(&ctx), (int64_t)seed));
  auto &funcBlock = func.getBody().emplaceBlock();
  builder.setInsertionPoint(&funcBlock, funcBlock.begin());

  // Create the generator info
  GeneratorInfo info(chooser, builder, availableOps, availableTypes,
                     availableAttributes, numArgs, createValueOutOfThinAir);

  auto type = availableTypes[chooser->choose(availableTypes.size())];
  auto root = info.addRootedOperation(type, numOps);
  if (!root.has_value())
    return nullptr;
  builder.create<func::ReturnOp>(unknownLoc, *root);
  func.insertResult(0, root->getType(), {});

  return module;
}