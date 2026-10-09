// LLVM integer operations on i3. i1 is only used where an operation requires
// it (icmp result, select condition, and as trunc/zext/sext counterpart).
// Use with: --configuration=llvm --int-widths=1,3
irdl.dialect @llvm {

  irdl.operation @add {
    %integer = irdl.is i3
    %ovf_none = irdl.is #llvm.overflow<none>
    %ovf_nsw = irdl.is #llvm.overflow<nsw>
    %ovf_nuw = irdl.is #llvm.overflow<nuw>
    %ovf_nsw_nuw = irdl.is #llvm.overflow<nsw,nuw>
    %ovf = irdl.any_of(%ovf_none, %ovf_nsw, %ovf_nuw, %ovf_nsw_nuw)
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
    irdl.attributes {"overflowFlags" = %ovf}
  }

  irdl.operation @sub {
    %integer = irdl.is i3
    %ovf_none = irdl.is #llvm.overflow<none>
    %ovf_nsw = irdl.is #llvm.overflow<nsw>
    %ovf_nuw = irdl.is #llvm.overflow<nuw>
    %ovf_nsw_nuw = irdl.is #llvm.overflow<nsw,nuw>
    %ovf = irdl.any_of(%ovf_none, %ovf_nsw, %ovf_nuw, %ovf_nsw_nuw)
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
    irdl.attributes {"overflowFlags" = %ovf}
  }

  irdl.operation @mul {
    %integer = irdl.is i3
    %ovf_none = irdl.is #llvm.overflow<none>
    %ovf_nsw = irdl.is #llvm.overflow<nsw>
    %ovf_nuw = irdl.is #llvm.overflow<nuw>
    %ovf_nsw_nuw = irdl.is #llvm.overflow<nsw,nuw>
    %ovf = irdl.any_of(%ovf_none, %ovf_nsw, %ovf_nuw, %ovf_nsw_nuw)
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
    irdl.attributes {"overflowFlags" = %ovf}
  }

  irdl.operation @udiv {
    %integer = irdl.is i3
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
    %unit = irdl.is unit
    irdl.attributes {"isExact" = %unit}
  }

  irdl.operation @sdiv {
    %integer = irdl.is i3
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
    %unit = irdl.is unit
    irdl.attributes {"isExact" = %unit}
  }

  irdl.operation @urem {
    %integer = irdl.is i3
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
  }

  irdl.operation @srem {
    %integer = irdl.is i3
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
  }

  irdl.operation @shl {
    %integer = irdl.is i3
    %ovf_none = irdl.is #llvm.overflow<none>
    %ovf_nsw = irdl.is #llvm.overflow<nsw>
    %ovf_nuw = irdl.is #llvm.overflow<nuw>
    %ovf_nsw_nuw = irdl.is #llvm.overflow<nsw,nuw>
    %ovf = irdl.any_of(%ovf_none, %ovf_nsw, %ovf_nuw, %ovf_nsw_nuw)
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
    irdl.attributes {"overflowFlags" = %ovf}
  }

  irdl.operation @lshr {
    %integer = irdl.is i3
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
    %unit = irdl.is unit
    irdl.attributes {"isExact" = %unit}
  }

  irdl.operation @ashr {
    %integer = irdl.is i3
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
    %unit = irdl.is unit
    irdl.attributes {"isExact" = %unit}
  }

  irdl.operation @and {
    %integer = irdl.is i3
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
  }

  irdl.operation @or {
    %integer = irdl.is i3
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
    %unit = irdl.is unit
    irdl.attributes {"isDisjoint" = %unit}
  }

  irdl.operation @xor {
    %integer = irdl.is i3
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
  }

  irdl.operation @trunc {
    %i1 = irdl.is i1
    %integer = irdl.is i3
    %ovf_none = irdl.is #llvm.overflow<none>
    %ovf_nsw = irdl.is #llvm.overflow<nsw>
    %ovf_nuw = irdl.is #llvm.overflow<nuw>
    %ovf_nsw_nuw = irdl.is #llvm.overflow<nsw,nuw>
    %ovf = irdl.any_of(%ovf_none, %ovf_nsw, %ovf_nuw, %ovf_nsw_nuw)
    irdl.operands(operand0: %integer)
    irdl.results(result0: %i1)
    irdl.attributes {"overflowFlags" = %ovf}
  }

  irdl.operation @zext {
    %i1 = irdl.is i1
    %integer = irdl.is i3
    irdl.operands(operand0: %i1)
    irdl.results(result0: %integer)
    %unit = irdl.is unit
    irdl.attributes {"nonNeg" = %unit}
  }

  irdl.operation @sext {
    %i1 = irdl.is i1
    %integer = irdl.is i3
    irdl.operands(operand0: %i1)
    irdl.results(result0: %integer)
  }

  irdl.operation @icmp {
    %integer = irdl.is i3
    %i1 = irdl.is i1
    // eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge
    %0 = irdl.is 0 : i64
    %1 = irdl.is 1 : i64
    %2 = irdl.is 2 : i64
    %3 = irdl.is 3 : i64
    %4 = irdl.is 4 : i64
    %5 = irdl.is 5 : i64
    %6 = irdl.is 6 : i64
    %7 = irdl.is 7 : i64
    %8 = irdl.is 8 : i64
    %9 = irdl.is 9 : i64
    %predT = irdl.any_of(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9)
    irdl.attributes {"predicate" = %predT}
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %i1)
  }

  irdl.operation @select {
    %cond = irdl.is i1
    %integer = irdl.is i3
    irdl.operands(operand0: %cond, operand1: %integer, operand2: %integer)
    irdl.results(result0: %integer)
  }

  irdl.operation @freeze {
    %integer = irdl.is i3
    irdl.operands(operand0: %integer)
    irdl.results(result0: %integer)
  }
}
