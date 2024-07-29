# Makefile

# Define the compiler and flags
CXX = clang++
CXXFLAGS = -std=c++17 -I/usr/include/llvm -I/usr/include/clang
LDFLAGS = -L/usr/lib/llvm -lLLVM -lclang

# Define the directories and files
SRC_DIR = src/compiler
BUILD_DIR = build
TARGET = $(BUILD_DIR)/parse

# Create the build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Rule to compile parse.cpp
$(TARGET): $(SRC_DIR)/parse.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/parse.cpp -o $(TARGET) $(LDFLAGS)

# Rule to clean the build directory
clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: clean
