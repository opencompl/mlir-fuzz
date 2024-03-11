get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        )

add_mlir_library(MLIRFuzzLib
  IRDLUtils.cpp
  GeneratorInfo.cpp
  CLITool.cpp
)
target_link_libraries(MLIRFuzzLib PRIVATE ${LIBS})