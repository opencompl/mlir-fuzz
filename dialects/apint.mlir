irdl.dialect @transfer {
    irdl.operation @and {
      %apint = irdl.is !transfer.integer
      irdl.operands(lhs: %apint, rhs: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @or {
      %apint = irdl.is !transfer.integer
      irdl.operands(lhs: %apint, rhs: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @xor {
      %apint = irdl.is !transfer.integer
      irdl.operands(lhs: %apint, rhs: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @add {
      %apint = irdl.is !transfer.integer
      irdl.operands(lhs: %apint, rhs: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @sub {
      %apint = irdl.is !transfer.integer
      irdl.operands(lhs: %apint, rhs: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @mul {
      %apint = irdl.is !transfer.integer
      irdl.operands(lhs: %apint, rhs: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @sdiv {
      %apint = irdl.is !transfer.integer
      irdl.operands(lhs: %apint, rhs: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @udiv {
      %apint = irdl.is !transfer.integer
      irdl.operands(lhs: %apint, rhs: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @srem {
      %apint = irdl.is !transfer.integer
      irdl.operands(lhs: %apint, rhs: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @urem {
      %apint = irdl.is !transfer.integer
      irdl.operands(lhs: %apint, rhs: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @shl {
      %apint = irdl.is !transfer.integer
      irdl.operands(value: %apint, shift: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @ashr {
      %apint = irdl.is !transfer.integer
      irdl.operands(value: %apint, shift: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @lshr {
      %apint = irdl.is !transfer.integer
      irdl.operands(value: %apint, shift: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @neg {
      %apint = irdl.is !transfer.integer
      irdl.operands(operand: %apint)
      irdl.results(result: %apint)
    }
    irdl.operation @smin {
      %apint = irdl.is !transfer.integer
      irdl.operands(lhs: %apint, rhs: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @smax {
      %apint = irdl.is !transfer.integer
      irdl.operands(lhs: %apint, rhs: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @umin {
      %apint = irdl.is !transfer.integer
      irdl.operands(lhs: %apint, rhs: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @umax {
      %apint = irdl.is !transfer.integer
      irdl.operands(lhs: %apint, rhs: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @countl_one {
      %apint = irdl.is !transfer.integer
      irdl.operands(operand: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @countl_zero {
      %apint = irdl.is !transfer.integer
      irdl.operands(operand: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @countr_one {
      %apint = irdl.is !transfer.integer
      irdl.operands(operand: %apint)
      irdl.results(result: %apint)
    }

    irdl.operation @countr_zero {
      %apint = irdl.is !transfer.integer
      irdl.operands(operand: %apint)
      irdl.results(result: %apint)
    }
}