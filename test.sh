original_dir=$(pwd)

g++ src/test.cpp -o src/test
./src/test

cd build && cmake ..
make
./ir
cd "$original_dir"