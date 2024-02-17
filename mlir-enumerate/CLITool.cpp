#include "CLITool.h"

#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;

std::vector<Type> getAvailableTypes(MLIRContext &ctx) {
  Builder builder(&ctx);
  return {builder.getIntegerType(1), builder.getIntegerType(8),
          builder.getIntegerType(32)};
}

std::vector<Attribute> getAvailableAttributes(MLIRContext &ctx) {
  Builder builder(&ctx);
  return {builder.getI64IntegerAttr(0), builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(2), builder.getI64IntegerAttr(3),
          builder.getI64IntegerAttr(4), builder.getI64IntegerAttr(5),
          builder.getI64IntegerAttr(6), builder.getI64IntegerAttr(7),
          builder.getI64IntegerAttr(8), builder.getI64IntegerAttr(9)};
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