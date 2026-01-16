module {
  irdl.dialect @riscv {

    irdl.operation @add {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    }

    irdl.operation @sub {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    }

    irdl.operation @xor {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    }

    irdl.operation @and {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    }

    irdl.operation @or attributes {synth} {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    }

    //irdl.operation @addi {
      //%register = irdl.is !riscv.reg
      //%0 = irdl.is 0 : si12 
      //%1 = irdl.is 1 : si12
      //%2 = irdl.is 2 : si12
      //%3 = irdl.is 3 : si12
      //%imm = irdl.any_of (%0, %1, %2, %3) 
      //irdl.operands(operand0: %register)
      //irdl.results(result0: %register)
      //irdl.attributes {"immediate" = %imm}
    //}

    //irdl.operation @xori {
      //%register = irdl.is !riscv.reg
      //%0 = irdl.is 0 : si12
      //%1 = irdl.is 1 : si12
      //%2 = irdl.is 2 : si12
      //%3 = irdl.is 3 : si12
      //%4 = irdl.is 4 : si12
      //%5 = irdl.is 5 : si12
      //%6 = irdl.is 6 : si12
      //%7 = irdl.is 7 : si12
      //%8 = irdl.is 8 : si12
      //%9 = irdl.is 9 : si12
      //%imm = irdl.any_of (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9) 
      //irdl.operands(operand0: %register)
      //irdl.results(result0: %register)
      //irdl.attributes {"immediate" = %imm}
    //}

    //irdl.operation @andi {
      //%register = irdl.is !riscv.reg
      //%0 = irdl.is 0 : si12
      //%1 = irdl.is 1 : si12
      //%2 = irdl.is 2 : si12
      //%3 = irdl.is 3 : si12
      //%4 = irdl.is 4 : si12
      //%5 = irdl.is 5 : si12
      //%6 = irdl.is 6 : si12
      //%7 = irdl.is 7 : si12
      //%8 = irdl.is 8 : si12
      //%9 = irdl.is 9 : si12
      //%imm = irdl.any_of (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9) 
      //irdl.operands(operand0: %register)
      //irdl.results(result0: %register)
      //irdl.attributes {"immediate" = %imm}
    //}

    //irdl.operation @ori {
      //%register = irdl.is !riscv.reg

      //%0 = irdl.is 0 : si12
      //%1 = irdl.is 1 : si12
      //%2 = irdl.is 2 : si12
      //%3 = irdl.is 3 : si12
      //%4 = irdl.is 4 : si12
      //%5 = irdl.is 5 : si12
      //%6 = irdl.is 6 : si12
      //%7 = irdl.is 7 : si12
      //%8 = irdl.is 8 : si12
      //%9 = irdl.is 9 : si12
      //%imm = irdl.any_of (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9) 
      //irdl.operands(operand0: %register)
      //irdl.results(result0: %register)
      //irdl.attributes {"immediate" = %imm}
    //}

    irdl.operation @slt {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    } 

  }
}