module {
    irdl.dialect @smt {
        irdl.operation @and {
            %bool = irdl.is !smt.bool
            irdl.operands(lhs: %bool, rhs: %bool)
            irdl.results(result: %bool)
        }

        irdl.operation @distinct {
            %bool = irdl.is !smt.bool
            %bv = irdl.base "!smt.bv"
            %t = irdl.any_of(%bool, %bv)
            irdl.operands(lhs: %t, rhs: %t)
            irdl.results(result: %bool)
        }

        irdl.operation @eq {
            %bool = irdl.is !smt.bool
            %bv = irdl.base "!smt.bv"
            %t = irdl.any_of(%bool, %bv)
            irdl.operands(lhs: %t, rhs: %t)
            irdl.results(result: %bool)
        }

        irdl.operation @implies {
            %bool = irdl.is !smt.bool
            irdl.operands(lhs: %bool, rhs: %bool)
            irdl.results(result: %bool)
        }

        irdl.operation @ite {
            %bool = irdl.is !smt.bool
            %bv = irdl.base "!smt.bv"
            %t = irdl.any_of(%bool, %bv)
            irdl.operands(cond: %bool, thenValue: %t, elseValue: %t)
            irdl.results(result: %t)
        }

        irdl.operation @not {
            %bool = irdl.is !smt.bool
            irdl.operands(input: %bool)
            irdl.results(result: %bool)
        }

        irdl.operation @or {
            %bool = irdl.is !smt.bool
            irdl.operands(lhs: %bool, rhs: %bool)
            irdl.results(result: %bool)
        }

        irdl.operation @xor {
            %bool = irdl.is !smt.bool
            irdl.operands(lhs: %bool, rhs: %bool)
            irdl.results(result: %bool)
        }
    }
}
