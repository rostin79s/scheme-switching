original_dir=$(pwd)

cd build && cmake ..
make
./main
cd "$original_dir"