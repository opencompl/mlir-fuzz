func.func @foo(%b: i32) -> i32 {
  %c = arith.addi %b, %b : i32
  %a = arith.addi %c, %b : i32
  %x = arith.addi %a, %b : i32
  %c32 = arith.constant 32 : i32
  %r = arith.muli %x, %c32 : i32
  func.return %r : i32
}
