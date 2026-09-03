CXX ?= clang++
CXXFLAGS ?= -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Werror
TESTFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror
CPPFLAGS ?= -Iinclude

BUILD_DIR := build
BENCH := $(BUILD_DIR)/gemm_bench
TESTS := $(BUILD_DIR)/tests
HEADERS := include/qik/quantize.hpp include/qik/gemm.hpp include/qik/gemm_neon.hpp

.PHONY: all test bench sanitize clean

all: $(BENCH) $(TESTS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BENCH): src/bench.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) src/bench.cpp -o $@

$(TESTS): tests/test_main.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(TESTFLAGS) tests/test_main.cpp -o $@

test: $(TESTS)
	./$(TESTS)

bench: $(BENCH)
	./$(BENCH)

sanitize: | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) -std=c++20 -O1 -g -Wall -Wextra -Wpedantic -Werror \
		-fsanitize=address,undefined -fno-omit-frame-pointer tests/test_main.cpp \
		-o $(BUILD_DIR)/tests_asan
	ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=print_stacktrace=1 ./$(BUILD_DIR)/tests_asan

clean:
	rm -rf $(BUILD_DIR)
