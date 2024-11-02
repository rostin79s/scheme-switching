original_dir=$(pwd)

# clang++ -S -emit-llvm -O3 src/test.cpp -o src/test.ll
cgeist test/test.cpp -S -O3 -raise-scf-to-affine > src/test.mlir
# mlir-opt --affine-super-vectorizer-test --vectorize-affine-loop-nest --vectorize-loops test.mlir -o output.mlir
cd src/build
cmake -G Ninja .. -DMLIR_DIR=/home/rostin/Polygeist/build/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=/home/rostin/Polygeist/build/bin/llvm-lit
cmake --build .

cd .. && ./compiler
/home/rostin/Polygeist/build/bin/mlir-translate -allow-unregistered-dialect --mlir-to-cpp ir.mlir -o ../test/output.cpp
cd "$original_dir"
cd test
g++ add_headers.cpp -o add_headers
./add_headers
rm add_headers