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

        irdl.operation @bv.add {
            %bv = irdl.base "!smt.bv"
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bv)
        }

        irdl.operation @bv.and {
            %bv = irdl.base "!smt.bv"
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bv)
        }

        irdl.operation @bv.ashr {
            %bv = irdl.base "!smt.bv"
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bv)
        }

        irdl.operation @bv.cmp {
            %bool = irdl.is !smt.bool
            %bv = irdl.base "!smt.bv"
            %c0_64 = irdl.is 0 : i64
            %c1_64 = irdl.is 1 : i64
            %c2_64 = irdl.is 2 : i64
            %c3_64 = irdl.is 3 : i64
            %c4_64 = irdl.is 4 : i64
            %c5_64 = irdl.is 5 : i64
            %c6_64 = irdl.is 6 : i64
            %c7_64 = irdl.is 7 : i64
            %predicate = irdl.any_of(
                %c0_64, %c1_64, %c2_64, %c3_64, %c4_64, %c5_64, %c6_64, %c7_64
            )
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bool)
            irdl.attributes {"pred" = %predicate}
        }

        // TODO: https://mlir.llvm.org/docs/Dialects/SMT/#smtbvconcat-mlirsmtconcatop
        // How to handle sizes properly?

        // TODO: Do we want that?
        // https://mlir.llvm.org/docs/Dialects/SMT/#smtbvconstant-mlirsmtbvconstantop
        // irdl.operation @bv.constant {
        //     %bv = irdl.base "!smt.bv"
        //     %0 = irdl.is 0
        //     %1 = irdl.is 1
        //     %value = irdl.any_of(%0, %1)
        //     irdl.attributes { "value" = %value }
        //     irdl.results(result: %bv)
        // }

        // TODO: https://mlir.llvm.org/docs/Dialects/SMT/#smtbvextract-mlirsmtextractop.
        // How to handle sizes properly?

        // irdl.operation @bv.lshr {
        //     %bv = irdl.base "!smt.bv"
        //     irdl.operands(lhs: %bv, rhs: %bv)
        //     irdl.results(result: %bv)
        // }

        irdl.operation @bv.mul {
            %bv = irdl.base "!smt.bv"
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bv)
        }

        // irdl.operation @bv.neg {
        //     %bv = irdl.base "!smt.bv"
        //     irdl.operands(input: %bv)
        //     irdl.results(result: %bv)
        // }

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

        // TODO: https://mlir.llvm.org/docs/Dialects/SMT/#smtbvrepeat-mlirsmtrepeatop

        // irdl.operation @bv.sdiv {
        //     %bv = irdl.base "!smt.bv"
        //     irdl.operands(lhs: %bv, rhs: %bv)
        //     irdl.results(result: %bv)
        // }

        // irdl.operation @bv.shl {
        //     %bv = irdl.base "!smt.bv"
        //     irdl.operands(lhs: %bv, rhs: %bv)
        //     irdl.results(result: %bv)
        // }

        // irdl.operation @bv.smod {
        //     %bv = irdl.base "!smt.bv"
        //     irdl.operands(lhs: %bv, rhs: %bv)
        //     irdl.results(result: %bv)
        // }

        // irdl.operation @bv.srem {
        //     %bv = irdl.base "!smt.bv"
        //     irdl.operands(lhs: %bv, rhs: %bv)
        //     irdl.results(result: %bv)
        // }

        // irdl.operation @bv.udiv {
        //     %bv = irdl.base "!smt.bv"
        //     irdl.operands(lhs: %bv, rhs: %bv)
        //     irdl.results(result: %bv)
        // }

        // irdl.operation @bv.urem {
        //     %bv = irdl.base "!smt.bv"
        //     irdl.operands(lhs: %bv, rhs: %bv)
        //     irdl.results(result: %bv)
        // }

        // irdl.operation @bv.xor {
        //     %bv = irdl.base "!smt.bv"
        //     irdl.operands(lhs: %bv, rhs: %bv)
        //     irdl.results(result: %bv)
        // }
    }
}
