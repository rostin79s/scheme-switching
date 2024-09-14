# Define variables
CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -I/usr/include/llvm -I/usr/local/include/mlir
LDFLAGS = -L/usr/include/llvm -L/usr/local/include/mlir -lLLVM -lMLIR

# Source files
SRCS = compiler.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Executable name
TARGET = compiler

# Default target
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $(TARGET) $(OBJS)

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
