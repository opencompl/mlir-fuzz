module {
    irdl.dialect @smt {
        irdl.operation @and {
            %bool = irdl.is !smt.bool
            irdl.operands(lhs: %bool, rhs: %bool)
            irdl.results(result: %bool)
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
