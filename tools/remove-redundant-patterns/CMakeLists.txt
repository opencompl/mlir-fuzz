add_llvm_executable(remove-redundant-patterns remove-redundant-patterns.cpp)
target_link_libraries(remove-redundant-patterns PRIVATE MLIRFuzzLib)

llvm_update_compile_flags(remove-redundant-patterns)

mlir_check_all_link_libraries(remove-redundant-patterns)
