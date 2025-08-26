#!/bin/bash
cd build
ninja mlir-enumerate
ninja superoptimizer
ninja check-subpattern
ninja remove-redundant-patterns
