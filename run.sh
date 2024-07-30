clang++ -S -emit-llvm -O3 ./src/test.cpp -o ./src/test.ll
clang++ -o ./src/test `llvm-config --cxxflags --ldflags --system-libs --libs core` ./src/compiler.cpp
cd src && ./test