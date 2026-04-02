# Compiler and flags
CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
INCLUDES = -Iinclude

# Source files and output
SRCDIR = src
SRCS   = $(SRCDIR)/main.cpp $(SRCDIR)/integrator.cpp
TARGET = build/ekf

# Default target
all: $(TARGET)

$(TARGET): $(SRCS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $(TARGET)

clean:
	rm -rf build

.PHONY: all clean