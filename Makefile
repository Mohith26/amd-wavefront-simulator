CXX = clang++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Werror

SRC = src/sim.cpp kernels/kernels.cpp
HDRS = src/ir.hpp src/sim.hpp kernels/kernels.hpp

all: build/tests build/bench

build/tests: tests/test_main.cpp $(SRC) $(HDRS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) tests/test_main.cpp $(SRC) -o $@

build/bench: bench/bench_main.cpp $(SRC) $(HDRS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) bench/bench_main.cpp $(SRC) -o $@

test: build/tests
	./build/tests

bench: build/bench
	@mkdir -p results
	./build/bench

clean:
	rm -rf build

.PHONY: all test bench clean
