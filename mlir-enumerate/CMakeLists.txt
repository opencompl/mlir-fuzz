get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRIRDL
        )
add_llvm_executable(mlir-enumerate mlir-enumerate.cpp)

llvm_update_compile_flags(mlir-enumerate)
target_link_libraries(mlir-enumerate PRIVATE ${LIBS})

mlir_check_all_link_libraries(mlir-enumerate)
