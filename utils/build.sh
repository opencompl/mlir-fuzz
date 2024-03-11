#!/bin/bash
cd build
ninja mlir-enumerate
ninja arith-superoptimizer
