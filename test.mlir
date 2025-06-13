module {
  func.func @main(%arg0: i64, %arg1: i1, %arg2: i64) -> i64 attributes {seed = 97 : index} {
    %0 = llvm.icmp "sle" %arg0, %arg0 : i64
    %1 = llvm.select %arg1, %arg2, %arg2 : i1, i64
    %2 = llvm.select %arg1, %1, %arg0 : i1, i64
    %3 = llvm.select %0, %2, %1 : i1, i64
    return %3 : i64
  }
}