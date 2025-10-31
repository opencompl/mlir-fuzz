irdl.dialect @llvm {

  irdl.operation @and {
    %integer = irdl.is i64
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
  }

  irdl.operation @or { // to do: need to add disjoint flag 
    %integer = irdl.is i64
    //%is_exact = irdl.is #llvm.isDisjoint
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
    %unit = irdl.is unit
    irdl.attributes {"isDisjointFlag" = %unit}
    // irdl.attributes {"isDisjoint" = %disjoint}
  }

  irdl.operation @xor { 
    %integer = irdl.is i64
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
  }

  irdl.operation @add { // to do: need to add llvm flags and not artih flags probably at CLITool.cpp
    %integer = irdl.is i64
    %ovf_none = irdl.is #llvm.overflow<none>
    %ovf_nsw = irdl.is #llvm.overflow<nsw>
    %ovf_nuw = irdl.is #llvm.overflow<nuw>
    %ovf_nsw_nuw = irdl.is #llvm.overflow<nsw,nuw>
    %ovf = irdl.any_of(%ovf_none, %ovf_nsw, %ovf_nuw, %ovf_nsw_nuw)
    irdl.operands(operand0: %integer, operand1: %integer)
    
    irdl.results(result0: %integer)
    irdl.attributes {"overflowFlags" = %ovf}
  }

  irdl.operation @sub { // to do: register correct flags aka do not use the arith flags 
    %integer = irdl.is i64
    %ovf_none = irdl.is #llvm.overflow<none>
    %ovf_nsw = irdl.is #llvm.overflow<nsw>
    %ovf_nuw = irdl.is #llvm.overflow<nuw>
    %ovf_nsw_nuw = irdl.is #llvm.overflow<nsw,nuw>
    %ovf = irdl.any_of(%ovf_none, %ovf_nsw, %ovf_nuw, %ovf_nsw_nuw)
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
    irdl.attributes {"overflowFlags" = %ovf}
  }

  irdl.operation @shl {  // to do: use the correct flags 
    %integer = irdl.is i64
    irdl.operands(operand0: %integer, operand1: %integer)
    %ovf_none = irdl.is #llvm.overflow<none>
    %ovf_nsw = irdl.is #llvm.overflow<nsw>
    %ovf_nuw = irdl.is #llvm.overflow<nuw>
    %ovf_nsw_nuw = irdl.is #llvm.overflow<nsw,nuw>
    %ovf = irdl.any_of(%ovf_none, %ovf_nsw, %ovf_nuw, %ovf_nsw_nuw)
    irdl.results(result0: %integer)
    irdl.attributes {"overflowFlags" = %ovf}
  }

  irdl.operation @lshr {  // to do : support is exact flag 
    %integer = irdl.is i64
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
    %unit = irdl.is unit
    irdl.attributes {"isExactFlag" = %unit}
  }

  irdl.operation @ashr {  // to do : support is exact flag 
    %integer = irdl.is i64
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
    %unit = irdl.is unit
    irdl.attributes {"isExactFlag" = %unit}
  }

  irdl.operation @mul { // to do : support llvm flags and not arith
    %integer = irdl.is i64
    irdl.operands(operand0: %integer, operand1: %integer)
    %ovf_none = irdl.is #llvm.overflow<none>
    %ovf_nsw = irdl.is #llvm.overflow<nsw>
    %ovf_nuw = irdl.is #llvm.overflow<nuw>
    %ovf_nsw_nuw = irdl.is #llvm.overflow<nsw,nuw>
    %ovf = irdl.any_of(%ovf_none, %ovf_nsw, %ovf_nuw, %ovf_nsw_nuw)
    irdl.results(result0: %integer)
    irdl.attributes {"overflowFlags" = %ovf}
  }

  irdl.operation @sdiv { // needs to isExact flag
    %integer = irdl.is i64
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
    %unit = irdl.is unit
    irdl.attributes {"isExactFlag" = %unit}
  }

  irdl.operation @udiv { // needs to isExact flag
    %integer = irdl.is i64
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
  }

  irdl.operation @urem { // needs to isExact flag
    %integer = irdl.is i64
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
  }

  irdl.operation @srem { // needs to isExact flag
    %integer = irdl.is i64
    irdl.operands(operand0: %integer, operand1: %integer)
    irdl.results(result0: %integer)
  }

  irdl.operation @icmp { // needs to isExact flag
    %integer64 = irdl.is i64 // restrict it to i64 bc do not want i1 atm.
    //%integer32 = irdl.is i32 : aka when supporting both i32 and i64
    //%integer = irdl.any_of(%integer64, %integer32)
    %i1 = irdl.is i1
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
    irdl.operands(operand0: %integer64, operand1: %integer64)
    irdl.results(result0: %i1)
  }

  irdl.operation @select { // needs to isExact flag
    %cond = irdl.is i1
    %integer = irdl.is i64
    irdl.operands(operand0: %cond, operand1: %integer, operand2:%integer)
    irdl.results(result0: %integer)
  }

  irdl.operation @trunc { // needs to isExact flag, supports conversion from i64 to i64, i32 and i1
    %integer1 = irdl.is i1
    %integer32 = irdl.is i32
    %integer = irdl.is i64
    %opType= irdl.any_of(%integer1, %integer32)
    irdl.operands(operand1: %integer)
    irdl.results(result0: %opType)
  }

  irdl.operation @sext { //supports conversion from i1 and i32 to i64
    %integer1 = irdl.is i1
    %integer32 = irdl.is i32
    %integer = irdl.is i64
    %opType= irdl.any_of(%integer1, %integer32)
    irdl.operands(operand1: %opType)
    irdl.results(result0: %integer)
  }

  irdl.operation @zext { //supports conversions from i1 and i32 to i64
    %integer1 = irdl.is i1
    %integer32 = irdl.is i32
    %integer = irdl.is i64
    %opType= irdl.any_of(%integer1, %integer32)
    irdl.operands(operand1: %opType)
    irdl.results(result0: %integer)
  }
}
