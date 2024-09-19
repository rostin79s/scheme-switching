original_dir=$(pwd)

# clang++ -S -emit-llvm -O3 src/test.cpp -o src/test.ll
cgeist test.cpp -S -O3 > src/test.mlir
cd src/build
cmake -G Ninja .. -DMLIR_DIR=/home/rostin/Polygeist/build/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=/home/rostin/Polygeist/build/bin/llvm-lit
cmake --build .

cd .. && ./compiler
/home/rostin/Polygeist/build/bin/mlir-translate -allow-unregistered-dialect --mlir-to-cpp ir.mlir -o ../output.cpp