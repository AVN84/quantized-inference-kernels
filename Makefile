CXX ?= clang++
CXXFLAGS ?= -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Werror
TESTFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror
CPPFLAGS ?= -Iinclude

# SMMLA (FEAT_I8MM) is not reachable from the default target on this hardware.
# The M3 Pro reports hw.optional.arm.FEAT_I8MM: 1, but clang's -mcpu=apple-m3
# model does not define __ARM_FEATURE_MATMUL_INT8, so a plain -O3 build silently
# gets SDOT and nothing else. -march=armv8.6-a+i8mm does define it.
#
# Two conditions have to hold before the flag goes in, and both are checked
# rather than assumed. The compiler has to accept the flag and define the
# feature macro, and the *host* has to actually implement the instruction --
# compiling SMMLA for a core without it produces a binary that builds cleanly
# and then dies on SIGILL. Probe both; if either fails, the build stays on SDOT
# and kHasNeonMatmulInt8 reports false.
I8MM_COMPILES := $(shell printf 'int main(){}' | $(CXX) -x c++ -std=c++20 -O1 \
	-march=armv8.6-a+i8mm -dM -E - 2>/dev/null \
	| grep -c __ARM_FEATURE_MATMUL_INT8)
I8MM_HOST := $(shell sysctl -n hw.optional.arm.FEAT_I8MM 2>/dev/null \
	|| grep -c -w i8mm /proc/cpuinfo 2>/dev/null || echo 0)

ifeq ($(I8MM_COMPILES),1)
ifneq ($(I8MM_HOST),0)
ARCHFLAGS := -march=armv8.6-a+i8mm
endif
endif

CXXFLAGS += $(ARCHFLAGS)
TESTFLAGS += $(ARCHFLAGS)

BUILD_DIR := build
BENCH := $(BUILD_DIR)/gemm_bench
TESTS := $(BUILD_DIR)/tests
METAL_BENCH := $(BUILD_DIR)/metal_bench
ACCURACY := $(BUILD_DIR)/accuracy
HEADERS := include/qik/quantize.hpp include/qik/quantize4.hpp include/qik/gemm.hpp include/qik/gemm_neon.hpp include/qik/gemm_smmla.hpp
OBJCXXFLAGS ?= -std=c++20 -ObjC++ -fobjc-arc -O2
METAL_LIBS := -framework Metal -framework Foundation

.PHONY: all test bench metal accuracy sanitize clean

all: $(BENCH) $(TESTS) $(ACCURACY)

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

$(ACCURACY): src/accuracy.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) src/accuracy.cpp -o $@

accuracy: $(ACCURACY)
	./$(ACCURACY)

sanitize: | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) -std=c++20 -O1 -g -Wall -Wextra -Wpedantic -Werror \
		$(ARCHFLAGS) -fsanitize=address,undefined -fno-omit-frame-pointer \
		tests/test_main.cpp -o $(BUILD_DIR)/tests_asan
	ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=print_stacktrace=1 ./$(BUILD_DIR)/tests_asan

clean:
	rm -rf $(BUILD_DIR)
