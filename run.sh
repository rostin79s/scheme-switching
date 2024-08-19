original_dir=$(pwd)

clang++ -S -emit-llvm -O3 ./src/test.cpp -o ./src/test.ll
make
cd src && ./test
cd "$original_dir"
make clean
