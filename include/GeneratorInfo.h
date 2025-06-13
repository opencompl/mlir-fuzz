//===- GeneratorInfo.h ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The fuzzing/enumerating context. It contains the list of operations
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_FUZZ_GENERATOR_INFO_H
#define MLIR_FUZZ_GENERATOR_INFO_H

#include "guide.h"
#include <functional>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"

namespace tree_guide {
class Chooser;
}

namespace mlir {
class ModuleOp;
}

/// Data structure to hold some information about the current program
/// being generated.
struct GeneratorInfo {
  /// The chooser, which will chose which path to take in the decision tree.
  tree_guide::Chooser *chooser;

  /// A builder set to the end of the function.
  mlir::OpBuilder builder;

  /// All available ops that can be used by the fuzzer.
  mlir::ArrayRef<mlir::irdl::OperationOp> availableOps;

  /// All available types that can be used by the fuzzer.
  mlir::ArrayRef<mlir::Type> availableTypes;

  /// All available attributes that can be used by the fuzzer.
  mlir::ArrayRef<mlir::Attribute> availableAttributes;

  /// The set of values that are dominating the insertion point.
  /// We group the values by their type.
  /// We store values of the same type in a vector to iterate on them
  /// deterministically.
  /// Since we are iterating from top to bottom of the program, we do not
  /// need to remove elements from this set.
  llvm::DenseMap<mlir::Type, std::vector<mlir::Value>> dominatingValues;

  /// The maximum number of arguments per function:
  int maxNumArgs;

  using CreateValueOutOfThinAirFn =
      std::function<std::optional<mlir::Value>(GeneratorInfo &, mlir::Type)>;

  /// Create a value from no other value.
  /// For instance, create a constant operation, or create a function argument.
  CreateValueOutOfThinAirFn createValueOutOfThinAir;

  GeneratorInfo(tree_guide::Chooser *chooser, mlir::OpBuilder builder,
                mlir::ArrayRef<mlir::irdl::OperationOp> availableOps,
                mlir::ArrayRef<mlir::Type> availableTypes,
                mlir::ArrayRef<mlir::Attribute> availableAttributes,
                int maxNumArgs,
                CreateValueOutOfThinAirFn createValueOutOfThinAir = nullptr);

  /// Add a value to the list of available values.
  void addDominatingValue(mlir::Value value) {
    dominatingValues[value.getType()].push_back(value);
  }

  /// Get a value in the program and its index in the dominating array.
  std::pair<std::optional<mlir::Value>, int> getValue(mlir::Type type) {
    auto &domValues = dominatingValues[type];

    if (domValues.size() == 0) {
      return {{}, -1};
    }
    auto choice = chooser->choose(domValues.size());
    return {domValues[choice], choice};
  }

  mlir::Value addFunctionArgument(mlir::Type type) {
    // Otherwise, add a new argument to the parent function.
    auto func = llvm::cast<mlir::func::FuncOp>(
        *builder.getInsertionBlock()->getParentOp());

    unsigned int position = func.getNumArguments();
    func.insertArgument(position, type, {},
                        mlir::UnknownLoc::get(builder.getContext()));
    auto arg = func.getArgument(position);
    addDominatingValue(arg);
    return arg;
  }

  /// Create a value of the given type, by materializing a constant.
  mlir::Value createIntegerValue(mlir::IntegerType type);

  /// Return the list of operations that can have a particular result type as
  /// result.
  /// Returns as well the indices of the results that can have this result type.
  std::vector<std::pair<mlir::irdl::OperationOp, std::vector<int>>>
  getOperationsWithResultType(mlir::Type resultType);

  /// Return the list of operations that can have a particular result type as
  /// result with a filter.
  /// We only consider operations making filter true.
  /// Returns as well the indices of the results that can have this result type.
  std::vector<std::pair<mlir::irdl::OperationOp, std::vector<int>>>
  getOperationsWithResultType(
      mlir::Type resultType,
      std::function<bool(mlir::irdl::OperationOp)> filter);

  /// Create an operation with a given operation op
  /// Return the operation created
  /// This function is used inside of addRootedOperation
  mlir::Operation *createOperation(mlir::irdl::OperationOp op,
                                   mlir::Type resultType, size_t resultIdx,
                                   int fuel);

  /// Add an operation with a given result type.
  /// Return the result that has has the requested type and the index of that
  /// value if it has zero cost. This function will also create a number
  /// proportional to `fuel` operations.
  std::pair<std::optional<mlir::Value>, int>
  addRootedOperation(mlir::Type resultType, int fuel);
};

/// Create a random program, given the decisions taken from chooser.
/// The program has at most `fuel` operations.
mlir::OwningOpRef<mlir::ModuleOp> createProgram(
    mlir::MLIRContext &ctx,
    mlir::ArrayRef<mlir::irdl::OperationOp> availableOps,
    mlir::ArrayRef<mlir::Type> availableTypes,
    mlir::ArrayRef<mlir::Attribute> availableAttributes,
    tree_guide::Chooser *chooser, int numOps, int numArgs, int seed,
    GeneratorInfo::CreateValueOutOfThinAirFn createValueOutOfThinAir = nullptr);

#endif // MLIR_FUZZ_GENERATOR_INFO_H