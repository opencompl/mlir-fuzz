#include "IRDLUtils.h"

using namespace mlir;
using namespace irdl;

/// Get the types that the constraint can support, given a constraint context.
std::vector<Type> getSatisfyingTypes(MLIRContext &ctx, int constraint,
                                     ConstraintVerifier &context,
                                     ArrayRef<Type> availableTypes) {
  std::vector<Type> satisfyingTypes;
  for (auto type : availableTypes) {
    ConstraintVerifier context_copy = context;
    if (context_copy.verify({}, TypeAttr::get(type), constraint).succeeded()) {
      satisfyingTypes.push_back(type);
    }
  }
  return satisfyingTypes;
}

std::pair<std::vector<std::unique_ptr<Constraint>>, DenseMap<Value, int>>
getOperationVerifier(OperationOp op) {
  // We do not handle dynamic tyes and attributes yet.
  DenseMap<TypeOp, std::unique_ptr<DynamicTypeDefinition>> types;
  DenseMap<AttributeOp, std::unique_ptr<DynamicAttrDefinition>> attrs;

  // Resolve SSA values to verifier constraint slots
  SmallVector<Value> constrToValue;
  DenseMap<Value, int> valueToIdx;
  for (Operation &op : op->getRegion(0).getOps()) {
    if (isa<VerifyConstraintInterface>(op)) {
      assert(op.getNumResults() == 1);
      valueToIdx[op.getResult(0)] = constrToValue.size();
      constrToValue.push_back(op.getResult(0));
    }
  }

  // Build the verifiers for each constraint slot
  std::vector<std::unique_ptr<Constraint>> constraints;
  DenseMap<Value, Constraint *> valueToConstraint;
  for (Value v : constrToValue) {
    VerifyConstraintInterface op =
        cast<VerifyConstraintInterface>(v.getDefiningOp());
    std::unique_ptr<Constraint> verifier =
        op.getVerifier(constrToValue, types, attrs);
    assert(verifier && "Constraint verifier couldn't be generated");
    valueToConstraint[v] = verifier.get();
    constraints.push_back(std::move(verifier));
  }

  return {std::move(constraints), std::move(valueToIdx)};
}