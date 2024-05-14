module {

// Not implemented because we don't use index: arith.index_cast, arith.index_castui
// Not implemented because we don't have constraints for integer attribute: arith.constant

  irdl.dialect @arith {


    irdl.operation @andi attributes {commutativity}{
    //irdl.operation @andi {
      %integer = irdl.base "!builtin.integer"
      irdl.operands(%integer, %integer)
      irdl.results(%integer)
    }

    //irdl.operation @xori {
    //  %integer = irdl.base "!builtin.integer"
    //  irdl.operands(%integer, %integer)
    //  irdl.results(%integer)
    //}
  }
}