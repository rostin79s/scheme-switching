original_dir=$(pwd)

clang++ -S -emit-llvm -O3 ./src/compiler/test.cpp -o ./src/compiler/test.ll
make
cd src/compiler && ./test
cd "$original_dir"
make clean
