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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"

namespace tree_guide {
class Chooser;
}

namespace mlir {
class ModuleOp;
}

/// Create a random program, given the decisions taken from chooser.
/// The program has at most `fuel` operations.
mlir::OwningOpRef<mlir::ModuleOp>
createProgram(mlir::MLIRContext &ctx,
              mlir::ArrayRef<mlir::irdl::OperationOp> availableOps,
              mlir::ArrayRef<mlir::Type> availableTypes,
              mlir::ArrayRef<mlir::Attribute> availableAttributes,
              tree_guide::Chooser *chooser, int numOps, int numArgs, int seed);

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

  GeneratorInfo(tree_guide::Chooser *chooser, mlir::OpBuilder builder,
                mlir::ArrayRef<mlir::irdl::OperationOp> availableOps,
                mlir::ArrayRef<mlir::Type> availableTypes,
                mlir::ArrayRef<mlir::Attribute> availableAttributes,
                int maxNumArgs)
      : chooser(chooser), builder(builder), availableOps(availableOps),
        availableTypes(availableTypes),
        availableAttributes(availableAttributes), maxNumArgs(maxNumArgs) {}

  /// Add a value to the list of available values.
  void addDominatingValue(mlir::Value value) {
    dominatingValues[value.getType()].push_back(value);
  }

  /// Get a value in the program.
  std::optional<mlir::Value> getValue(mlir::Type type) {
    auto &domValues = dominatingValues[type];

    if (domValues.size() == 0) {
      return {};
    }
    auto choice = chooser->choose(domValues.size());
    return domValues[choice];
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
  std::optional<mlir::Value> createValueOutOfThinAir(mlir::Type type);

  /// Return the list of operations that can have a particular result type as
  /// result.
  /// Returns as well the indices of the results that can have this result type.
  std::vector<std::pair<mlir::irdl::OperationOp, std::vector<int>>>
  getOperationsWithResultType(mlir::Type resultType);

  /// Add an operation with a given result type.
  /// Return the result that has has the requested type.
  /// This function will also create a number proportional to `fuel` operations.
  std::optional<mlir::Value> addRootedOperation(mlir::Type resultType,
                                                int fuel);
};

#endif // MLIR_FUZZ_GENERATOR_INFO_H