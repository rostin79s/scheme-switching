clang++ -S -emit-llvm -O3 test.cpp -o test.ll
clang++ -o test `llvm-config --cxxflags --ldflags --system-libs --libs core` compiler.cpp
./test