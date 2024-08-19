# Compiler and flags
CXX = clang++
LLVM_CONFIG = llvm-config
CXXFLAGS = `$(LLVM_CONFIG) --cxxflags`
LDFLAGS = `$(LLVM_CONFIG) --ldflags --system-libs --libs core`

# Directories and files
SRCDIR = src
BUILDDIR = src
TARGET = $(BUILDDIR)/test

# Source files
SOURCES = $(SRCDIR)/compiler.cpp $(SRCDIR)/frontend/frontend.cpp $(SRCDIR)/frontend/dag.cpp $(SRCDIR)/backend/backend.cpp
OBJECTS = $(patsubst $(SRCDIR)/%.cpp, $(BUILDDIR)/%.o, $(SOURCES))

# Default target
all: $(TARGET)

# Custom clang++ command to generate src/compiler/test
$(TARGET): $(SRCDIR)/compiler.cpp $(OBJECTS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(SOURCES)

# Rule to compile object files
$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

# Clean up build files
clean:
	rm -f $(BUILDDIR)/*.o $(BUILDDIR)/frontend/*.o $(BUILDDIR)/backend/*.o
