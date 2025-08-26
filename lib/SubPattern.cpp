#include "SubPattern.h"

using namespace mlir;
using namespace mlir::func;

bool match(Value lhs, Value rhs, std::vector<Value> &mapping) {
  if (lhs == rhs)
    return true;

  if (lhs.getType() != rhs.getType())
    return false;

  // Every value can match a block argument (a free variable).
  // If we try to match against a block argument twice, we check that
  // both values we tried to match are equal.
  if (auto lhsArg = dyn_cast<BlockArgument>(lhs)) {
    Value &mappingArg = mapping[(size_t)lhsArg.getArgNumber()];
    if (mappingArg)
      // We use pointer equality, as we assume that the program is cse'd.
      return mappingArg == rhs;
    mappingArg = rhs;
    return true;
  }

  auto lhsRes = dyn_cast<OpResult>(lhs);
  auto rhsRes = dyn_cast<OpResult>(rhs);

  if (!lhsRes || !rhsRes)
    return false;

  if (lhsRes.getResultNumber() != rhsRes.getResultNumber())
    return false;

  Operation *lhsOp = lhsRes.getOwner();
  Operation *rhsOp = rhsRes.getOwner();
  if (lhsRes.getOwner()->getName() != rhsRes.getOwner()->getName())
    return false;

  if (lhsOp->getAttrDictionary() != rhsOp->getAttrDictionary())
    return false;

  if (lhsOp->getNumOperands() != rhsOp->getNumOperands())
    return false;

  for (auto [lhsOperand, rhsOperand] :
       llvm::zip(lhsOp->getOperands(), rhsOp->getOperands())) {
    if (!match(lhsOperand, rhsOperand, mapping))
      return false;
  }
  return true;
}

bool isSubPattern(FuncOp lhs, FuncOp rhs) {
  if (lhs.getResultTypes().size() != rhs.getResultTypes().size())
    return false;

  ReturnOp lhsRet = cast<ReturnOp>(lhs.getBody().front().getTerminator());
  ReturnOp rhsRet = cast<ReturnOp>(rhs.getBody().front().getTerminator());

  std::vector<Value> mapping(lhs.getArgumentTypes().size(), nullptr);

  for (auto [lhsArg, rhsArg] :
       llvm::zip(lhsRet.getOperands(), rhsRet.getOperands())) {
    if (!match(lhsArg, rhsArg, mapping))
      return false;
  }
  return true;
}