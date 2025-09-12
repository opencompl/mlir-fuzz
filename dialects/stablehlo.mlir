irdl.dialect @stablehlo {
    irdl.operation @add {
        %tensor = irdl.base "!builtin.tensor"
        irdl.operands(lhs: %tensor, rhs: %tensor)
        irdl.results(result: %tensor)
    }
}