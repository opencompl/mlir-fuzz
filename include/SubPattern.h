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

#ifndef IRDL_SUBPATTERNS_H
#define IRDL_SUBPATTERNS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

/// Return true if lhs is a subpattern of rhs.
bool isSubPattern(mlir::func::FuncOp lhs, mlir::func::FuncOp rhs);

#endif // IRDL_SUBPATTERNS_H