#include <cstring>
#import <Metal/Metal.h>

#include <string>
#include <vector>

#include "qik/metal_gemm.hpp"

namespace qik {
namespace {

// Compiled at runtime rather than shipped as a .metallib. The offline metal
// compiler ships with full Xcode, not the Command Line Tools, so precompiling
// would make this repo un-buildable on a machine that has the Metal framework
// and a working GPU but no Xcode. Runtime compilation costs a few milliseconds
// once and is paid outside the timed region.
constexpr const char* kShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

// One thread per output element. Each walks a row of A against a row of Bt,
// which is the same access pattern the CPU kernel uses -- both operands stream
// contiguously. Accumulation is int32 and therefore exact, so this must agree
// with the scalar CPU kernel bit for bit, not approximately.
kernel void gemm_int8(device const char*  A    [[buffer(0)]],
                      device const char*  Bt   [[buffer(1)]],
                      device int*         C    [[buffer(2)]],
                      constant uint3&     dims [[buffer(3)]],
                      uint2               gid  [[thread_position_in_grid]])
{
    const uint M = dims.x, N = dims.y, K = dims.z;
    const uint n = gid.x, m = gid.y;
    if (m >= M || n >= N) { return; }

    device const char* a = A  + (ulong)m * K;
    device const char* b = Bt + (ulong)n * K;

    int sum = 0;
    uint k = 0;
    // Four at a time. The widening to int happens before the multiply, so a
    // product of two -128..127 values cannot overflow the accumulator.
    for (; k + 4 <= K; k += 4) {
        sum += int(a[k + 0]) * int(b[k + 0]);
        sum += int(a[k + 1]) * int(b[k + 1]);
        sum += int(a[k + 2]) * int(b[k + 2]);
        sum += int(a[k + 3]) * int(b[k + 3]);
    }
    for (; k < K; ++k) { sum += int(a[k]) * int(b[k]); }

    C[(ulong)m * N + n] = sum;
}
)MSL";

struct MetalContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline = nil;
  std::string error;
  bool ok = false;
};

// Built once and reused. Rebuilding the pipeline per call would dominate the
// measurement of anything short.
MetalContext& context() {
  static MetalContext ctx = [] {
    MetalContext c;
    @autoreleasepool {
      // MTLCreateSystemDefaultDevice returns nil without a window server
      // session, which is exactly what happens over SSH or in a build agent.
      // MTLCopyAllDevices still enumerates the GPU, so prefer the default and
      // fall back rather than reporting "no GPU" on a machine that has one.
      id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
      if (dev == nil) {
        NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
        if ([all count] > 0) { dev = all[0]; }
      }
      if (dev == nil) { c.error = "no Metal device available"; return c; }
      c.device = dev;

      NSError* err = nil;
      id<MTLLibrary> lib =
          [dev newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource]
                            options:nil
                              error:&err];
      if (lib == nil) {
        c.error = "shader compile failed: " +
                  std::string([[err localizedDescription] UTF8String]);
        return c;
      }
      id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_int8"];
      if (fn == nil) { c.error = "kernel gemm_int8 not found"; return c; }

      c.pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
      if (c.pipeline == nil) {
        c.error = "pipeline creation failed";
        return c;
      }
      c.queue = [dev newCommandQueue];
      if (c.queue == nil) { c.error = "command queue creation failed"; return c; }
      c.ok = true;
    }
    return c;
  }();
  return ctx;
}

}  // namespace

bool metal_available(std::string* error) {
  MetalContext& c = context();
  if (!c.ok && error != nullptr) { *error = c.error; }
  return c.ok;
}

std::string metal_device_name() {
  MetalContext& c = context();
  if (!c.ok) { return "none"; }
  @autoreleasepool {
    return std::string([[c.device name] UTF8String]);
  }
}

bool gemm_int8_metal(const std::int8_t* a, const std::int8_t* bt,
                     std::int32_t* c_out, int m_dim, int n_dim, int k_dim,
                     std::string* error) {
  MetalContext& ctx = context();
  if (!ctx.ok) {
    if (error != nullptr) { *error = ctx.error; }
    return false;
  }
  if (m_dim <= 0 || n_dim <= 0 || k_dim <= 0) {
    if (error != nullptr) { *error = "dimensions must be positive"; }
    return false;
  }

  @autoreleasepool {
    const size_t a_bytes = (size_t)m_dim * k_dim;
    const size_t b_bytes = (size_t)n_dim * k_dim;
    const size_t c_bytes = (size_t)m_dim * n_dim * sizeof(std::int32_t);

    // Shared storage: unified memory, so these are views rather than copies.
    id<MTLBuffer> ba = [ctx.device newBufferWithBytes:a
                                               length:a_bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [ctx.device newBufferWithBytes:bt
                                               length:b_bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> bc = [ctx.device newBufferWithLength:c_bytes
                                               options:MTLResourceStorageModeShared];
    if (ba == nil || bb == nil || bc == nil) {
      if (error != nullptr) { *error = "buffer allocation failed"; }
      return false;
    }

    const uint32_t dims[3] = {(uint32_t)m_dim, (uint32_t)n_dim, (uint32_t)k_dim};

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.pipeline];
    [enc setBuffer:ba offset:0 atIndex:0];
    [enc setBuffer:bb offset:0 atIndex:1];
    [enc setBuffer:bc offset:0 atIndex:2];
    [enc setBytes:dims length:sizeof(dims) atIndex:3];

    // Threadgroup shaped to the hardware SIMD width rather than a round number,
    // so a group maps onto whole SIMD groups instead of straddling them.
    const NSUInteger width = [ctx.pipeline threadExecutionWidth];
    const NSUInteger max_threads = [ctx.pipeline maxTotalThreadsPerThreadgroup];
    NSUInteger tg_x = width;
    NSUInteger tg_y = max_threads / width;
    if (tg_y == 0) { tg_y = 1; }

    [enc dispatchThreads:MTLSizeMake((NSUInteger)n_dim, (NSUInteger)m_dim, 1)
      threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if ([cb status] == MTLCommandBufferStatusError) {
      if (error != nullptr) {
        NSError* e = [cb error];
        *error = "command buffer failed: " +
                 std::string(e ? [[e localizedDescription] UTF8String] : "unknown");
      }
      return false;
    }

    std::memcpy(c_out, [bc contents], c_bytes);
  }
  return true;
}

}  // namespace qik
