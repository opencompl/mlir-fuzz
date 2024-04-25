func.func @foo(%x: i32) -> i32 {
  %c32 = arith.constant 32 : i32
  %r = arith.muli %x, %c32 : i32
  return %x : i32
}
