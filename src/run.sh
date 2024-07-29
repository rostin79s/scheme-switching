clang++ -o test `llvm-config --cxxflags --ldflags --system-libs --libs core` compiler.cpp
./test
