#include "IRDLUtils.h"

using namespace mlir;
using namespace irdl;

/// Get the attributes that the constraint can support, given a constraint
/// context.
std::vector<Attribute> getSatisfyingAttrs(MLIRContext &ctx, int constraint,
                                          ConstraintVerifier &context,
                                          ArrayRef<Attribute> availableAttrs) {
  std::vector<Attribute> satisfyingAttrs;
  for (auto attr : availableAttrs) {
    ConstraintVerifier context_copy = context;
    if (context_copy.verify({}, attr, constraint).succeeded()) {
      satisfyingAttrs.push_back(attr);
    }
  }
  return satisfyingAttrs;
}

std::vector<Attribute> getSatisfyingAttrs(MLIRContext &ctx, Value value,
                                          OperationOp op,
                                          ArrayRef<Attribute> availableAttrs) {
  auto [constraints, valueToIdx] = getOperationVerifier(op);
  ConstraintVerifier verifier(constraints);
  return getSatisfyingAttrs(ctx, valueToIdx[value], verifier, availableAttrs);
}

/// Get the types that the constraint can support, given a constraint context.
std::vector<Type> getSatisfyingTypes(MLIRContext &ctx, int constraint,
                                     ConstraintVerifier &context,
                                     ArrayRef<Type> availableTypes) {
  std::vector<Attribute> availableAttrs;
  for (auto type : availableTypes) {
    availableAttrs.push_back(TypeAttr::get(type));
  }
  auto res = getSatisfyingAttrs(ctx, constraint, context, availableAttrs);
  std::vector<Type> resType;
  for (auto attr : res) {
    resType.push_back(attr.cast<TypeAttr>().getValue());
  }
  return resType;
}

std::vector<Type> getSatisfyingTypes(MLIRContext &ctx, Value value,
                                     OperationOp op,
                                     ArrayRef<Type> availableTypes) {
  auto [constraints, valueToIdx] = getOperationVerifier(op);
  ConstraintVerifier verifier(constraints);
  return getSatisfyingTypes(ctx, valueToIdx[value], verifier, availableTypes);
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

std::vector<Value> getOperandsConstraints(OperationOp op) {
  auto operandsOp = op.getOp<OperandsOp>();
  if (!operandsOp)
    return {};
  std::vector<Value> res;
  res.insert(res.begin(), operandsOp->getOperands().begin(),
             operandsOp->getOperands().end());
  return res;
}

/// Get the result constraints as a list of values.
std::vector<mlir::Value> getResultsConstraints(mlir::irdl::OperationOp op) {
  auto resultsOp = op.getOp<ResultsOp>();
  if (!resultsOp)
    return {};
  std::vector<Value> res;
  res.insert(res.begin(), resultsOp->getOperands().begin(),
             resultsOp->getOperands().end());
  return res;
}

std::vector<std::pair<StringRef, mlir::Value>>
getAttributesConstraints(mlir::irdl::OperationOp op) {
  auto attrOp = op.getOp<AttributesOp>();
  if (!attrOp)
    return {};
  auto attrNames = attrOp->getAttributeValueNames();
  auto attrValues = attrOp->getAttributeValues();

  std::vector<std::pair<StringRef, mlir::Value>> res;
  for (size_t i = 0; i < attrNames.size(); i++)
    res.emplace_back(attrNames[i].cast<StringAttr>().getValue(), attrValues[i]);
  return res;
}
