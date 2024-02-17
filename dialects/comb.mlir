module {

// Not implemented because we don't have constraints for integer attribute: extract, replicate, truth_table

  irdl.dialect @comb {
    irdl.operation @add {
      %integer = irdl.base "!builtin.integer"
      irdl.operands(%integer, %integer)
      irdl.results(%integer)
    }

    irdl.operation @and {
      %integer = irdl.base "!builtin.integer"
      irdl.operands(%integer, %integer)
      irdl.results(%integer)
    }

    irdl.operation @divs {
      %integer = irdl.base "!builtin.integer"
      irdl.operands(%integer, %integer)
      irdl.results(%integer)
    }

    irdl.operation @divu {
      %integer = irdl.base "!builtin.integer"
      irdl.operands(%integer, %integer)
      irdl.results(%integer)
    }

    irdl.operation @icmp {
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
      %10 = irdl.is 10 : i64
      %11 = irdl.is 11 : i64
      %12 = irdl.is 12 : i64
      %13 = irdl.is 13 : i64

      %predicate = irdl.any_of(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)

      %integer = irdl.base "!builtin.integer"
      %i1 = irdl.is i1

      irdl.operands(%integer, %integer)
      irdl.results(%i1)
      irdl.attributes { "predicate" = %predicate }
    }

    irdl.operation @mods {
      %integer = irdl.base "!builtin.integer"
      irdl.operands(%integer, %integer)
      irdl.results(%integer)
    }

    irdl.operation @modu {
      %integer = irdl.base "!builtin.integer"
      irdl.operands(%integer, %integer)
      irdl.results(%integer)
    }

    irdl.operation @mul {
      %integer = irdl.base "!builtin.integer"
      irdl.operands(%integer, %integer)
      irdl.results(%integer)
    }

    irdl.operation @mux {
      %i1 = irdl.is i1
      %integer = irdl.base "!builtin.integer"
      irdl.operands(%i1, %integer, %integer)
      irdl.results(%integer)
    }

    irdl.operation @or {
      %integer = irdl.base "!builtin.integer"
      irdl.operands(%integer, %integer)
      irdl.results(%integer)
    }

    irdl.operation @parity {
      %integer = irdl.base "!builtin.integer"
      %i1 = irdl.is i1
      irdl.operands(%integer)
      irdl.results(%i1)
    }

    irdl.operation @shl {
      %integer = irdl.base "!builtin.integer"
      irdl.operands(%integer, %integer)
      irdl.results(%integer)
    }

    irdl.operation @shrs {
      %integer = irdl.base "!builtin.integer"
      irdl.operands(%integer, %integer)
      irdl.results(%integer)
    }

    irdl.operation @shru {
      %integer = irdl.base "!builtin.integer"
      irdl.operands(%integer, %integer)
      irdl.results(%integer)
    }

    irdl.operation @sub {
      %integer = irdl.base "!builtin.integer"
      irdl.operands(%integer, %integer)
      irdl.results(%integer)
    }

    irdl.operation @xor {
      %integer = irdl.base "!builtin.integer"
      irdl.operands(%integer, %integer)
      irdl.results(%integer)
    }
  }
}