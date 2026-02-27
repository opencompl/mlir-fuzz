module {
  irdl.dialect @riscv {

    irdl.operation @add {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    }

    irdl.operation @addi attributes {synth} {
      %register = irdl.is !riscv.reg
      //For synth ops attributes will be constrained to 0, 
      //then later generalized by the fuzzer to any value of their type
      %imm = irdl.is 0 : si12 
      irdl.operands(operand0: %register)
      irdl.results(result0: %register)
      irdl.attributes {"immediate" = %imm}
    }

    irdl.operation @addw {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    }

    //Disabled due to not being in xdsl riscv dialect
    //irdl.operation @addiw attributes {synth} {
      //%register = irdl.is !riscv.reg
      //%imm = irdl.is 0 : si12 
      //irdl.operands(operand0: %register)
      //irdl.results(result0: %register)
      //irdl.attributes {"immediate" = %imm}
    //}

    irdl.operation @sub {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    }

    irdl.operation @subw {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    }

    irdl.operation @xor {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    }

    irdl.operation @xori attributes {synth} {
      %register = irdl.is !riscv.reg
      %imm = irdl.is 0 : si12
      irdl.operands(operand0: %register)
      irdl.results(result0: %register)
      irdl.attributes {"immediate" = %imm}
    }

    irdl.operation @and {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    }

    irdl.operation @andi attributes {synth} {
      %register = irdl.is !riscv.reg
      %imm = irdl.is 0 : si12
      irdl.operands(operand0: %register)
      irdl.results(result0: %register)
      irdl.attributes {"immediate" = %imm}
    }

    irdl.operation @or {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    }

    irdl.operation @ori attributes {synth} {
      %register = irdl.is !riscv.reg
      %imm = irdl.is 0 : si12
      irdl.operands(operand0: %register)
      irdl.results(result0: %register)
      irdl.attributes {"immediate" = %imm}
    }

    irdl.operation @slt {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    } 

    irdl.operation @slti attributes {synth} {
      %register = irdl.is !riscv.reg
      %imm = irdl.is 0 : si12
      irdl.operands(operand0: %register)
      irdl.results(result0: %register)
      irdl.attributes {"immediate" = %imm}
    }

    irdl.operation @sltu {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    } 

    irdl.operation @sltiu attributes {synth} {
      %register = irdl.is !riscv.reg
      %imm = irdl.is 0 : si12
      irdl.operands(operand0: %register)
      irdl.results(result0: %register)
      irdl.attributes {"immediate" = %imm}
    }

    irdl.operation @sll {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    } 

    irdl.operation @slli attributes {synth} {
      %register = irdl.is !riscv.reg
      %imm = irdl.is 0 : ui5
      irdl.operands(operand0: %register)
      irdl.results(result0: %register)
      irdl.attributes {"immediate" = %imm}
    }

    irdl.operation @sllw {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    } 

    //irdl.operation @slliw attributes {synth} {
      //%register = irdl.is !riscv.reg
      //%imm = irdl.is 0 : si12
      //irdl.operands(operand0: %register)
      //irdl.results(result0: %register)
      //irdl.attributes {"immediate" = %imm}
    //}

    //irdl.operation @slli.wu attributes {synth} {
      //%register = irdl.is !riscv.reg
      //%imm = irdl.is 0 : si12
      //irdl.operands(operand0: %register)
      //irdl.results(result0: %register)
      //irdl.attributes {"immediate" = %imm}
    //}

    irdl.operation @srl {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    } 

    irdl.operation @srli attributes {synth} {
      %register = irdl.is !riscv.reg
      %imm = irdl.is 0 : ui5
      irdl.operands(operand0: %register)
      irdl.results(result0: %register)
      irdl.attributes {"immediate" = %imm}
    }

    irdl.operation @srlw {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    } 

    irdl.operation @srliw attributes {synth} {
      %register = irdl.is !riscv.reg
      %imm = irdl.is 0 : ui5
      irdl.operands(operand0: %register)
      irdl.results(result0: %register)
      irdl.attributes {"immediate" = %imm}
    }

    irdl.operation @sra {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    } 

    irdl.operation @srai attributes {synth} {
      %register = irdl.is !riscv.reg
      %imm = irdl.is 0 : ui5
      irdl.operands(operand0: %register)
      irdl.results(result0: %register)
      irdl.attributes {"immediate" = %imm}
    }

    irdl.operation @sraw {
      %register = irdl.is !riscv.reg
      irdl.operands(operand0: %register, operand1: %register)
      irdl.results(result0: %register)
    } 

    irdl.operation @sraiw attributes {synth} {
      %register = irdl.is !riscv.reg
      %imm = irdl.is 0 : ui5
      irdl.operands(operand0: %register)
      irdl.results(result0: %register)
      irdl.attributes {"immediate" = %imm}
    }
  }
}
