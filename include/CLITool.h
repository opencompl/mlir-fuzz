//===- CLITool.h ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Some functions that are useful for CLI tools.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_FUZZ_CLI_TOOL_H
#define MLIR_FUZZ_CLI_TOOL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

enum class Configuration {
  Arith,
  Comb,
  SMT,
};

/// Get the types that the fuzzer supports.
std::vector<mlir::Type> getAvailableTypes(mlir::MLIRContext &ctx,
                                          Configuration config,
                                          std::vector<unsigned> smtBvWidths);

/// Get the types that the fuzzer supports.
std::vector<mlir::Attribute> getAvailableAttributes(mlir::MLIRContext &ctx,
                                                    Configuration config);

/// Parse an MLIR file given a filename.
std::optional<mlir::OwningOpRef<mlir::ModuleOp>>
parseMLIRFile(mlir::MLIRContext &ctx, mlir::StringRef inputFilename);

#endif // MLIR_FUZZ_CLI_TOOL_H
