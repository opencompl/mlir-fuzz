module {
    irdl.dialect @smt {
        irdl.operation @or {
            %bool = irdl.is !smt.bool
            irdl.operands(inputs: variadic %bool)
            irdl.results(result: %bool)
        }
    }
}