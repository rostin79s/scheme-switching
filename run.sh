original_dir=$(pwd)

# g++ test.cpp -o test
# ./test

cd build 
cmake -DNATIVE_SIZE=32 -DWITH_NATIVEOPT=ON -DCMAKE_C_COMPILER=clang-12 -DCMAKE_CXX_COMPILER=clang++-12 -DWITH_OPENMP=OFF ..
make
# ./ir
./execute
# ./main
# ./example
cd "$original_dir"