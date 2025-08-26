module {
    irdl.dialect @smt {
        irdl.operation @bv.add {
            %bv = irdl.base "!smt.bv"
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bv)
        }

        irdl.operation @bv.sub {
            %bv = irdl.base "!smt.bv"
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bv)
        }

        irdl.operation @bv.and {
            %bv = irdl.base "!smt.bv"
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bv)
        }

        irdl.operation @bv.lshr {
            %bv = irdl.base "!smt.bv"
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bv)
        }

        irdl.operation @bv.mul {
            %bv = irdl.base "!smt.bv"
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bv)
        }

        irdl.operation @bv.neg {
            %bv = irdl.base "!smt.bv"
            irdl.operands(input: %bv)
            irdl.results(result: %bv)
        }

        irdl.operation @bv.not {
            %bv = irdl.base "!smt.bv"
            irdl.operands(input: %bv)
            irdl.results(result: %bv)
        }

        irdl.operation @bv.or {
            %bv = irdl.base "!smt.bv"
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bv)
        }

        irdl.operation @bv.shl {
            %bv = irdl.base "!smt.bv"
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bv)
        }
    }
}
