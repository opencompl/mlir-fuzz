add_llvm_executable(mlir-enumerate mlir-enumerate.cpp)
target_link_libraries(mlir-enumerate PRIVATE MLIRFuzzLib)

llvm_update_compile_flags(mlir-enumerate)

mlir_check_all_link_libraries(mlir-enumerate)
