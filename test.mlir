"builtin.module"() ({
  "func.func"() <{function_type = (!riscv.reg) -> !riscv.reg, sym_name = "main"}> ({
  ^bb0(%arg0: !riscv.reg):
    %0 = "riscv.addi"(%arg0) <{immediate = 0 : si12}> : (!riscv.reg) -> !riscv.reg
    "func.return"(%0) : (!riscv.reg) -> ()
  }) : () -> ()
}) : () -> ()