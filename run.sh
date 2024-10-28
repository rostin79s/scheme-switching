original_dir=$(pwd)

# g++ test.cpp -o test
# ./test

cd build && cmake ..
make
# ./ir
./main
# ./example
cd "$original_dir"