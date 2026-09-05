CXX ?= clang++
CXXFLAGS ?= -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Werror
TESTFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror
CPPFLAGS ?= -Iinclude

BUILD_DIR := build
BENCH := $(BUILD_DIR)/gemm_bench
TESTS := $(BUILD_DIR)/tests
METAL_BENCH := $(BUILD_DIR)/metal_bench
HEADERS := include/qik/quantize.hpp include/qik/gemm.hpp include/qik/gemm_neon.hpp
OBJCXXFLAGS ?= -std=c++20 -ObjC++ -fobjc-arc -O2
METAL_LIBS := -framework Metal -framework Foundation

.PHONY: all test bench metal sanitize clean

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

# Metal is macOS only and needs the framework, so it is a separate target
# rather than part of `all`. The shader is compiled at runtime, so this does
# not require a full Xcode install -- only the Metal framework and a GPU.
$(METAL_BENCH): src/metal_bench.cpp src/metal_gemm.mm include/qik/metal_gemm.hpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c src/metal_bench.cpp -o $(BUILD_DIR)/metal_bench.o
	$(CXX) $(CPPFLAGS) $(OBJCXXFLAGS) -c src/metal_gemm.mm -o $(BUILD_DIR)/metal_gemm.o
	$(CXX) $(BUILD_DIR)/metal_bench.o $(BUILD_DIR)/metal_gemm.o $(METAL_LIBS) -o $@

metal: $(METAL_BENCH)
	./$(METAL_BENCH)

sanitize: | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) -std=c++20 -O1 -g -Wall -Wextra -Wpedantic -Werror \
		-fsanitize=address,undefined -fno-omit-frame-pointer tests/test_main.cpp \
		-o $(BUILD_DIR)/tests_asan
	ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=print_stacktrace=1 ./$(BUILD_DIR)/tests_asan

clean:
	rm -rf $(BUILD_DIR)
