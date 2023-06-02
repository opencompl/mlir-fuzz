//===- graph.h --------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <unordered_map>
#include <unordered_set>

using NodeID = std::int64_t;

/// A rule is a set of input types and outputs.
/// Input types are represented by the integer bitwidth.
struct Rule {
  std::unordered_multiset<int> inputs;
  int output;
};

struct NodeInfo {
  int distance;
  std::unordered_map<NodeID, Rule> nextNodes;
};

//
struct Graph {
  std::unordered_map<NodeID, NodeID> nodes;
};
