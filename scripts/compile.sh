original_dir=$(pwd)

# clang++ -S -emit-llvm -O3 src/test.cpp -o src/test.ll
cgeist test/test.cpp -S -O3 -raise-scf-to-affine --polyhedral-opt > src/frontend/test.mlir
mlir-opt -affine-super-vectorize="virtual-vector-size=5 test-fastest-varying=0 vectorize-reductions=true" src/frontend/test.mlir -o src/frontend/output.mlir
mlir-opt -lower-affine src/frontend/output.mlir -o src/frontend/output1.mlir


cd src/build
cmake -G Ninja .. -DMLIR_DIR=/home/rostin/Polygeist/build/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=/home/rostin/Polygeist/build/bin/llvm-lit
cmake --build .
cd .. && ./compiler


/home/rostin/Polygeist/build/bin/mlir-translate -allow-unregistered-dialect --mlir-to-cpp frontend/ir.mlir -o ../test/output.cpp


cd "$original_dir"
cd test
g++ add_headers.cpp -o add_headers
./add_headers
rm add_headers