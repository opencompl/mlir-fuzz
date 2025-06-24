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

        // TODO: https://mlir.llvm.org/docs/Dialects/SMT/#smtbvcmp-mlirsmtbvcmpop

        // TODO: https://mlir.llvm.org/docs/Dialects/SMT/#smtbvconcat-mlirsmtconcatop

        // TODO: https://mlir.llvm.org/docs/Dialects/SMT/#smtbvconstant-mlirsmtbvconstantop

        // TODO: https://mlir.llvm.org/docs/Dialects/SMT/#smtbvextract-mlirsmtextractop.

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

        // TODO: https://mlir.llvm.org/docs/Dialects/SMT/#smtbvrepeat-mlirsmtrepeatop

        irdl.operation @bv.sdiv {
            %bv = irdl.base "!smt.bv"
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bv)
        }

        irdl.operation @bv.shl {
            %bv = irdl.base "!smt.bv"
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bv)
        }

        irdl.operation @bv.smod {
            %bv = irdl.base "!smt.bv"
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bv)
        }

        irdl.operation @bv.srem {
            %bv = irdl.base "!smt.bv"
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bv)
        }

        irdl.operation @bv.udiv {
            %bv = irdl.base "!smt.bv"
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bv)
        }

        irdl.operation @bv.urem {
            %bv = irdl.base "!smt.bv"
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bv)
        }

        irdl.operation @bv.xor {
            %bv = irdl.base "!smt.bv"
            irdl.operands(lhs: %bv, rhs: %bv)
            irdl.results(result: %bv)
        }
    }
}
