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
#include "mlir/Dialect/IRDL/IRDLContext.h"
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

  /// Context for the runtime registration of IRDL dialect definitions.
  mlir::irdl::IRDLContext &irdlContext;

  /// The set of values that are dominating the insertion point.
  /// We group the values by their type.
  /// We store values of the same type in a vector to iterate on them
  /// deterministically.
  /// Since we are iterating from top to bottom of the program, we do not
  /// need to remove elements from this set.
  llvm::DenseMap<mlir::Type, std::vector<mlir::Value>> dominatingValues;

  GeneratorInfo(tree_guide::Chooser *chooser,
                mlir::ArrayRef<mlir::irdl::OperationOp> availableOps,
                mlir::OpBuilder builder, mlir::irdl::IRDLContext &irdlContext)
      : chooser(chooser), availableOps(availableOps), builder(builder),
        irdlContext(irdlContext) {}

  /// Add a value to the list of available values.
  void addDominatingValue(mlir::Value value) {
    dominatingValues[value.getType()].push_back(value);
  }
};

/// Get a value in the program.
/// This may add a new argument to the function.
inline mlir::Value getValue(GeneratorInfo &info, mlir::Type type) {
  auto builder = info.builder;
  auto &domValues = info.dominatingValues[type];

  // For now, we assume that we are only generating values of the same type.
  auto choice = info.chooser->choose(domValues.size() + 1);

  // If we chose a dominating value, return it
  if (choice < (long)domValues.size()) {
    return domValues[choice];
  }

  // Otherwise, add a new argument to the parent function.
  auto func = llvm::cast<mlir::func::FuncOp>(
      *builder.getInsertionBlock()->getParentOp());

  // We first chose an index where to add this argument.
  // Note that this is very costly when we are enumerating all programs of
  // a certain size.
  auto newArgIndex = info.chooser->choose(func.getNumArguments() + 1);

  func.insertArgument(newArgIndex, type, {},
                      mlir::UnknownLoc::get(builder.getContext()));
  auto arg = func.getArgument(newArgIndex);
  info.addDominatingValue(arg);
  return arg;
}

#endif // MLIR_FUZZ_GENERATOR_INFO_H