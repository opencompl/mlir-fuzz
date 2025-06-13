
cmake --build build --target mlir-enumerate                                     
./build/bin/mlir-enumerate dialects/llvm.mlir --max-num-ops=100 --configuration=llvm --strategy=bfs --max-programs=100