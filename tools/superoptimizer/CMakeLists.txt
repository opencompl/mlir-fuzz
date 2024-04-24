set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
  )

add_llvm_executable(superoptimizer superoptimizer.cpp)

target_link_libraries(superoptimizer
  PRIVATE
  MLIRFuzzLib
  MLIRExecutionEngine
  MLIRJitRunner
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRToLLVMIRTranslationRegistration
  MLIRTargetLLVMIRExport
)

llvm_update_compile_flags(superoptimizer)

mlir_check_all_link_libraries(superoptimizer)
