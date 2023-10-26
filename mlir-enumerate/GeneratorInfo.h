//===- tblgen-extract.cpp --------------------------------------*- C++ -*-===//
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
#include "mlir/Support/LLVM.h"

/// Data structure to hold some information about the current program
/// being generated.
struct GeneratorInfo {
  /// The chooser, which will chose which path to take in the decision tree.
  tree_guide::Chooser *chooser;

  /// All available ops that can be used by the fuzzer.
  mlir::ArrayRef<mlir::irdl::OperationOp> availableOps;

  /// A builder set to the end of the function.
  mlir::OpBuilder builder;

  /// The set of values that are dominating the insertion point.
  /// We group the values by their type.
  /// We store values of the same type in a vector to iterate on them
  /// deterministically.
  /// Since we are iterating from top to bottom of the program, we do not
  /// need to remove elements from this set.
  llvm::DenseMap<mlir::Type, std::vector<mlir::Value>> dominatingValues;

  GeneratorInfo(tree_guide::Chooser *chooser,
                mlir::ArrayRef<mlir::irdl::OperationOp> availableOps,
                mlir::OpBuilder builder)
      : chooser(chooser), availableOps(availableOps), builder(builder) {}

  /// Add a value to the list of available values.
  void addDominatingValue(mlir::Value value) {
    dominatingValues[value.getType()].push_back(value);
  }

  /// Get a value in the program.
  /// This may add a new argument to the function.
  std::optional<mlir::Value> getValue(mlir::Type type, bool addAsArgument) {
    auto &domValues = dominatingValues[type];

    if (domValues.size() + addAsArgument == 0) {
      return {};
    }
    // For now, we assume that we are only generating values of the same type.
    auto choice = chooser->choose(domValues.size() + addAsArgument);

    // If we chose a dominating value, return it
    if (choice < (long)domValues.size()) {
      return domValues[choice];
    }

    return addFunctionArgument(type);
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
};

#endif // MLIR_FUZZ_GENERATOR_INFO_H