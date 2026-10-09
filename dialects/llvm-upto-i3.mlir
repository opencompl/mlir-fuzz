// LLVM integer operations on all integer types up to i3 (i1, i2, i3).
// Binary operations, icmp, select, and freeze use a single width per
// operation; trunc, zext, and sext have independent source and result widths.
// Use with: --configuration=llvm --int-widths=1,2,3
irdl.dialect @llvm {

  irdl.operation @add {
    %int_i1 = irdl.is i1
    %int_i2 = irdl.is i2
    %int_i3 = irdl.is i3
    %int = irdl.any_of(%int_i1, %int_i2, %int_i3)
    %ovf_none = irdl.is #llvm.overflow<none>
    %ovf_nsw = irdl.is #llvm.overflow<nsw>
    %ovf_nuw = irdl.is #llvm.overflow<nuw>
    %ovf_nsw_nuw = irdl.is #llvm.overflow<nsw,nuw>
    %ovf = irdl.any_of(%ovf_none, %ovf_nsw, %ovf_nuw, %ovf_nsw_nuw)
    irdl.operands(operand0: %int, operand1: %int)
    irdl.results(result0: %int)
    irdl.attributes {"overflowFlags" = %ovf}
  }

  irdl.operation @sub {
    %int_i1 = irdl.is i1
    %int_i2 = irdl.is i2
    %int_i3 = irdl.is i3
    %int = irdl.any_of(%int_i1, %int_i2, %int_i3)
    %ovf_none = irdl.is #llvm.overflow<none>
    %ovf_nsw = irdl.is #llvm.overflow<nsw>
    %ovf_nuw = irdl.is #llvm.overflow<nuw>
    %ovf_nsw_nuw = irdl.is #llvm.overflow<nsw,nuw>
    %ovf = irdl.any_of(%ovf_none, %ovf_nsw, %ovf_nuw, %ovf_nsw_nuw)
    irdl.operands(operand0: %int, operand1: %int)
    irdl.results(result0: %int)
    irdl.attributes {"overflowFlags" = %ovf}
  }

  irdl.operation @mul {
    %int_i1 = irdl.is i1
    %int_i2 = irdl.is i2
    %int_i3 = irdl.is i3
    %int = irdl.any_of(%int_i1, %int_i2, %int_i3)
    %ovf_none = irdl.is #llvm.overflow<none>
    %ovf_nsw = irdl.is #llvm.overflow<nsw>
    %ovf_nuw = irdl.is #llvm.overflow<nuw>
    %ovf_nsw_nuw = irdl.is #llvm.overflow<nsw,nuw>
    %ovf = irdl.any_of(%ovf_none, %ovf_nsw, %ovf_nuw, %ovf_nsw_nuw)
    irdl.operands(operand0: %int, operand1: %int)
    irdl.results(result0: %int)
    irdl.attributes {"overflowFlags" = %ovf}
  }

  irdl.operation @udiv {
    %int_i1 = irdl.is i1
    %int_i2 = irdl.is i2
    %int_i3 = irdl.is i3
    %int = irdl.any_of(%int_i1, %int_i2, %int_i3)
    %unit = irdl.is unit
    irdl.operands(operand0: %int, operand1: %int)
    irdl.results(result0: %int)
    irdl.attributes {"isExact" = %unit}
  }

  irdl.operation @sdiv {
    %int_i1 = irdl.is i1
    %int_i2 = irdl.is i2
    %int_i3 = irdl.is i3
    %int = irdl.any_of(%int_i1, %int_i2, %int_i3)
    %unit = irdl.is unit
    irdl.operands(operand0: %int, operand1: %int)
    irdl.results(result0: %int)
    irdl.attributes {"isExact" = %unit}
  }

  irdl.operation @urem {
    %int_i1 = irdl.is i1
    %int_i2 = irdl.is i2
    %int_i3 = irdl.is i3
    %int = irdl.any_of(%int_i1, %int_i2, %int_i3)
    irdl.operands(operand0: %int, operand1: %int)
    irdl.results(result0: %int)
  }

  irdl.operation @srem {
    %int_i1 = irdl.is i1
    %int_i2 = irdl.is i2
    %int_i3 = irdl.is i3
    %int = irdl.any_of(%int_i1, %int_i2, %int_i3)
    irdl.operands(operand0: %int, operand1: %int)
    irdl.results(result0: %int)
  }

  irdl.operation @shl {
    %int_i1 = irdl.is i1
    %int_i2 = irdl.is i2
    %int_i3 = irdl.is i3
    %int = irdl.any_of(%int_i1, %int_i2, %int_i3)
    %ovf_none = irdl.is #llvm.overflow<none>
    %ovf_nsw = irdl.is #llvm.overflow<nsw>
    %ovf_nuw = irdl.is #llvm.overflow<nuw>
    %ovf_nsw_nuw = irdl.is #llvm.overflow<nsw,nuw>
    %ovf = irdl.any_of(%ovf_none, %ovf_nsw, %ovf_nuw, %ovf_nsw_nuw)
    irdl.operands(operand0: %int, operand1: %int)
    irdl.results(result0: %int)
    irdl.attributes {"overflowFlags" = %ovf}
  }

  irdl.operation @lshr {
    %int_i1 = irdl.is i1
    %int_i2 = irdl.is i2
    %int_i3 = irdl.is i3
    %int = irdl.any_of(%int_i1, %int_i2, %int_i3)
    %unit = irdl.is unit
    irdl.operands(operand0: %int, operand1: %int)
    irdl.results(result0: %int)
    irdl.attributes {"isExact" = %unit}
  }

  irdl.operation @ashr {
    %int_i1 = irdl.is i1
    %int_i2 = irdl.is i2
    %int_i3 = irdl.is i3
    %int = irdl.any_of(%int_i1, %int_i2, %int_i3)
    %unit = irdl.is unit
    irdl.operands(operand0: %int, operand1: %int)
    irdl.results(result0: %int)
    irdl.attributes {"isExact" = %unit}
  }

  irdl.operation @and {
    %int_i1 = irdl.is i1
    %int_i2 = irdl.is i2
    %int_i3 = irdl.is i3
    %int = irdl.any_of(%int_i1, %int_i2, %int_i3)
    irdl.operands(operand0: %int, operand1: %int)
    irdl.results(result0: %int)
  }

  irdl.operation @or {
    %int_i1 = irdl.is i1
    %int_i2 = irdl.is i2
    %int_i3 = irdl.is i3
    %int = irdl.any_of(%int_i1, %int_i2, %int_i3)
    %unit = irdl.is unit
    irdl.operands(operand0: %int, operand1: %int)
    irdl.results(result0: %int)
    irdl.attributes {"isDisjoint" = %unit}
  }

  irdl.operation @xor {
    %int_i1 = irdl.is i1
    %int_i2 = irdl.is i2
    %int_i3 = irdl.is i3
    %int = irdl.any_of(%int_i1, %int_i2, %int_i3)
    irdl.operands(operand0: %int, operand1: %int)
    irdl.results(result0: %int)
  }

  irdl.operation @trunc {
    %src_i1 = irdl.is i1
    %src_i2 = irdl.is i2
    %src_i3 = irdl.is i3
    %src = irdl.any_of(%src_i1, %src_i2, %src_i3)
    %dst_i1 = irdl.is i1
    %dst_i2 = irdl.is i2
    %dst_i3 = irdl.is i3
    %dst = irdl.any_of(%dst_i1, %dst_i2, %dst_i3)
    %ovf_none = irdl.is #llvm.overflow<none>
    %ovf_nsw = irdl.is #llvm.overflow<nsw>
    %ovf_nuw = irdl.is #llvm.overflow<nuw>
    %ovf_nsw_nuw = irdl.is #llvm.overflow<nsw,nuw>
    %ovf = irdl.any_of(%ovf_none, %ovf_nsw, %ovf_nuw, %ovf_nsw_nuw)
    irdl.operands(operand0: %src)
    irdl.results(result0: %dst)
    irdl.attributes {"overflowFlags" = %ovf}
  }

  irdl.operation @zext {
    %src_i1 = irdl.is i1
    %src_i2 = irdl.is i2
    %src_i3 = irdl.is i3
    %src = irdl.any_of(%src_i1, %src_i2, %src_i3)
    %dst_i1 = irdl.is i1
    %dst_i2 = irdl.is i2
    %dst_i3 = irdl.is i3
    %dst = irdl.any_of(%dst_i1, %dst_i2, %dst_i3)
    %unit = irdl.is unit
    irdl.operands(operand0: %src)
    irdl.results(result0: %dst)
    irdl.attributes {"nonNeg" = %unit}
  }

  irdl.operation @sext {
    %src_i1 = irdl.is i1
    %src_i2 = irdl.is i2
    %src_i3 = irdl.is i3
    %src = irdl.any_of(%src_i1, %src_i2, %src_i3)
    %dst_i1 = irdl.is i1
    %dst_i2 = irdl.is i2
    %dst_i3 = irdl.is i3
    %dst = irdl.any_of(%dst_i1, %dst_i2, %dst_i3)
    irdl.operands(operand0: %src)
    irdl.results(result0: %dst)
  }

  irdl.operation @icmp {
    %int_i1 = irdl.is i1
    %int_i2 = irdl.is i2
    %int_i3 = irdl.is i3
    %int = irdl.any_of(%int_i1, %int_i2, %int_i3)
    %i1 = irdl.is i1
    // eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge
    %p0 = irdl.is 0 : i64
    %p1 = irdl.is 1 : i64
    %p2 = irdl.is 2 : i64
    %p3 = irdl.is 3 : i64
    %p4 = irdl.is 4 : i64
    %p5 = irdl.is 5 : i64
    %p6 = irdl.is 6 : i64
    %p7 = irdl.is 7 : i64
    %p8 = irdl.is 8 : i64
    %p9 = irdl.is 9 : i64
    %pred = irdl.any_of(%p0, %p1, %p2, %p3, %p4, %p5, %p6, %p7, %p8, %p9)
    irdl.attributes {"predicate" = %pred}
    irdl.operands(operand0: %int, operand1: %int)
    irdl.results(result0: %i1)
  }

  irdl.operation @select {
    %int_i1 = irdl.is i1
    %int_i2 = irdl.is i2
    %int_i3 = irdl.is i3
    %int = irdl.any_of(%int_i1, %int_i2, %int_i3)
    %cond = irdl.is i1
    irdl.operands(operand0: %cond, operand1: %int, operand2: %int)
    irdl.results(result0: %int)
  }

  irdl.operation @freeze {
    %int_i1 = irdl.is i1
    %int_i2 = irdl.is i2
    %int_i3 = irdl.is i3
    %int = irdl.any_of(%int_i1, %int_i2, %int_i3)
    irdl.operands(operand0: %int)
    irdl.results(result0: %int)
  }
}
