irdl.dialect @transfer {
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
}