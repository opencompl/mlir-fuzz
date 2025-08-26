add_llvm_executable(check-subpattern check-subpattern.cpp)
target_link_libraries(check-subpattern PRIVATE MLIRFuzzLib)

llvm_update_compile_flags(check-subpattern)

mlir_check_all_link_libraries(check-subpattern)
