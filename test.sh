original_dir=$(pwd)

g++ test.cpp -o test
./src/test

cd build && cmake ..
make
./ir
./main
cd "$original_dir"