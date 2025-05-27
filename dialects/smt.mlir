module {
    irdl.dialect @smt {
        irdl.operation @or {
            %bool = irdl.is !smt.bool
            irdl.operands(lhs: %bool, rhs: %bool)
            irdl.results(result: %bool)
        }
    }
}