#include "CLITool.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Dialect/SMT/IR/SMTTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;

std::vector<Type> getAvailableTypes(MLIRContext &ctx, Configuration config,
                                    std::vector<unsigned> smtBvWidths) {
  Builder builder(&ctx);
  switch (config) {
  case Configuration::Arith:
  case Configuration::Comb:
    return {
        builder.getIntegerType(1),
        builder.getIntegerType(8),
        builder.getIntegerType(32),
        builder.getIntegerType(64),
    };
  case Configuration::SMT: {
    std::vector<Type> types = {smt::BoolType::get(&ctx)};
    for (unsigned width : smtBvWidths) {
      types.push_back(smt::BitVectorType::get(&ctx, width));
    }
    return types;
  }
  case Configuration::LLVM:
    return {
        builder.getIntegerType(1),
        builder.getIntegerType(8),
        builder.getIntegerType(64),
    };
  case Configuration::Tensor:
    return {
        builder.getF32Type(),
        builder.getIndexType(),
        builder.getIntegerType(64),
        RankedTensorType::get({2, 2}, builder.getF32Type()),
        RankedTensorType::get({4, 4}, builder.getF32Type()),
        RankedTensorType::get({2, 4}, builder.getF32Type()),
        RankedTensorType::get({4, 2}, builder.getF32Type()),
    };
  };
  llvm_unreachable("Unknown configuration");
}

std::vector<Attribute> getAvailableAttributes(MLIRContext &ctx,
                                              Configuration config) {
  Builder builder(&ctx);
  switch (config) {
  case Configuration::Arith:
  case Configuration::Comb:
    return {builder.getUnitAttr(),
            builder.getI64IntegerAttr(0),
            builder.getI64IntegerAttr(1),
            builder.getI64IntegerAttr(2),
            builder.getI64IntegerAttr(3),
            builder.getI64IntegerAttr(4),
            builder.getI64IntegerAttr(5),
            builder.getI64IntegerAttr(6),
            builder.getI64IntegerAttr(7),
            builder.getI64IntegerAttr(8),
            builder.getI64IntegerAttr(9),
            arith::IntegerOverflowFlagsAttr::get(
                &ctx, arith::IntegerOverflowFlags::none),
            arith::IntegerOverflowFlagsAttr::get(
                &ctx, arith::IntegerOverflowFlags::nsw),
            arith::IntegerOverflowFlagsAttr::get(
                &ctx, arith::IntegerOverflowFlags::nuw),
            arith::IntegerOverflowFlagsAttr::get(
                &ctx, arith::IntegerOverflowFlags::nsw |
                          arith::IntegerOverflowFlags::nuw)};
  case Configuration::SMT:
    return {
        builder.getUnitAttr(),
        IntegerAttr::get(builder.getIntegerType(1), -1),
        IntegerAttr::get(builder.getIntegerType(1), 0),
        builder.getI64IntegerAttr(0),
        builder.getI64IntegerAttr(1),
        builder.getI64IntegerAttr(2),
        builder.getI64IntegerAttr(3),
        builder.getI64IntegerAttr(4),
        builder.getI64IntegerAttr(5),
        builder.getI64IntegerAttr(6),
        builder.getI64IntegerAttr(7),
    };
  case Configuration::LLVM:
    return {builder.getI64IntegerAttr(0),
            builder.getI64IntegerAttr(1),
            builder.getI64IntegerAttr(2),
            builder.getI64IntegerAttr(3),
            builder.getI64IntegerAttr(4),
            builder.getI64IntegerAttr(5),
            builder.getI64IntegerAttr(6),
            builder.getI64IntegerAttr(7),
            builder.getI64IntegerAttr(8),
            builder.getI64IntegerAttr(9),
            builder.getUnitAttr(), // For 'exact'
            builder.getUnitAttr(), // For 'disjoint'
            LLVM::IntegerOverflowFlagsAttr::get(
                &ctx, LLVM::IntegerOverflowFlags::none),
            LLVM::IntegerOverflowFlagsAttr::get(
                &ctx, LLVM::IntegerOverflowFlags::nsw),
            LLVM::IntegerOverflowFlagsAttr::get(
                &ctx, LLVM::IntegerOverflowFlags::nuw),
            LLVM::IntegerOverflowFlagsAttr::get(
                &ctx, LLVM::IntegerOverflowFlags::nsw |
                          LLVM::IntegerOverflowFlags::nuw)};
  case Configuration::Tensor:
    return {};
  }
  llvm_unreachable("Unknown configuration");
}

std::optional<OwningOpRef<ModuleOp>> parseMLIRFile(MLIRContext &ctx,
                                                   StringRef inputFilename) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return std::nullopt;
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
