set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
  )

add_llvm_executable(arith-superoptimizer arith-superoptimizer.cpp)

target_link_libraries(arith-superoptimizer
  PRIVATE
  MLIRFuzzLib
  MLIRExecutionEngine
  MLIRJitRunner
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRToLLVMIRTranslationRegistration
  MLIRTargetLLVMIRExport
)

llvm_update_compile_flags(arith-superoptimizer)

mlir_check_all_link_libraries(arith-superoptimizer)
