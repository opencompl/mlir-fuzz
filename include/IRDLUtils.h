//===- IRDLUtils.h ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provide utils for IRDL defined operations.
//
//===----------------------------------------------------------------------===//

#ifndef IRDL_UTILS_H
#define IRDL_UTILS_H

#include "mlir/IR/Dialect.h"
#include <vector>

#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IRDLVerifiers.h"

/// Get the attributes that the constraint can support, given a constraint
/// context.
std::vector<mlir::Attribute>
getSatisfyingAttrs(mlir::MLIRContext &ctx, int constraint,
                   mlir::irdl::ConstraintVerifier &context,
                   mlir::ArrayRef<mlir::Attribute> availableAttrs);

/// Get the attributes that the constraint can support, given a constraint
/// context.
std::vector<mlir::Attribute>
getSatisfyingAttrs(mlir::MLIRContext &ctx, mlir::Value value,
                   mlir::irdl::OperationOp op,
                   mlir::ArrayRef<mlir::Attribute> availableAttrs);

/// Get the types that the constraint can support, given a constraint context.
std::vector<mlir::Type>
getSatisfyingTypes(mlir::MLIRContext &ctx, int constraint,
                   mlir::irdl::ConstraintVerifier &context,
                   mlir::ArrayRef<mlir::Type> availableTypes);

/// Get the types that a given value in an irdl operation can support.
std::vector<mlir::Type>
getSatisfyingTypes(mlir::MLIRContext &ctx, mlir::Value value,
                   mlir::irdl::OperationOp op,
                   mlir::ArrayRef<mlir::Type> availableTypes);

/// Get the IRDL constraint verifier from an Operation.
std::pair<std::vector<std::unique_ptr<mlir::irdl::Constraint>>,
          mlir::DenseMap<mlir::Value, int>>
getOperationVerifier(mlir::irdl::OperationOp op);

/// Get the operands constraints as a list of values.
std::vector<mlir::Value> getOperandsConstraints(mlir::irdl::OperationOp op);

/// Get the result constraints as a list of values.
std::vector<mlir::Value> getResultsConstraints(mlir::irdl::OperationOp op);

/// Get the attribute constraints as a list of names and values.
std::vector<std::pair<mlir::StringRef, mlir::Value>>
getAttributesConstraints(mlir::irdl::OperationOp op);

#endif // IRDL_UTILS_H