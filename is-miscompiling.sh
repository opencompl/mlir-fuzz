#!/usr/bin/env bash
mlir-opt $1 -o $1.before.tmp --mlir-print-op-generic
mlir-opt $1.before.tmp --arith-expand --mlir-print-op-generic -o $1.after.tmp
xdsl-tv $1.before.tmp $1.after.tmp | z3 -in > $1.z3res.tmp
! grep -q unsat $1.z3res.tmp
