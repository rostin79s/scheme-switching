original_dir=$(pwd)
cd build 
pwd
make clean
cmake -DNATIVE_SIZE=32 -DCMAKE_C_COMPILER=clang-12 -DCMAKE_CXX_COMPILER=clang++-12 -DWITH_NTL=ON -DOMP_NUM_THREADS=24 ..
make

# ./ir
# ./execute
# ./main
# ./example
cd "$original_dir"