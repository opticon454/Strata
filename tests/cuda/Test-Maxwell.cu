// Bounded correctness checks for pre-Pascal arithmetic; no model or timing loops.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "strata/kernels/dp4a.hpp"

struct Input { int32_t a, b, c; float x, y, z; };
struct Output { int32_t dot; float half_lo, half_hi, fused; };

__global__ void Check_Arithmetic(const Input* input, Output* output, size_t count) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const Input v = input[i];
    const half2 a = __floats2half2_rn(v.x, -v.x);
    const half2 b = __floats2half2_rn(v.y, v.y);
    const half2 c = __floats2half2_rn(v.z, -v.z);
    const float2 h = __half22float2(__hadd2(__hmul2(a, b), c));
    output[i] = {STRATA_DP4A(v.a, v.b, v.c), h.x, h.y, fmaf(v.x, v.y, v.z)};
}

static int Signed_Byte(uint32_t value, int lane) {
    const int byte = (value >> (8 * lane)) & 255;
    return byte < 128 ? byte : byte - 256;
}

#define CHECK_CUDA(call) do { const cudaError_t error = (call); \
    if (error != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); return 1; } } while (0)

int main() {
    int devices = 0;
    const cudaError_t available = cudaGetDeviceCount(&devices);
    if (available == cudaErrorNoDevice || (available == cudaSuccess && devices == 0)) {
        std::puts("SKIP: no CUDA device; hardware correctness was not checked");
        return 77;
    }
    CHECK_CUDA(available);
    cudaDeviceProp device{};
    CHECK_CUDA(cudaGetDeviceProperties(&device, 0));
    std::printf("device: %s (sm_%d%d)\n", device.name, device.major, device.minor);
    // Every signed-byte pair in every lane, with edge bytes in the other lanes.
    constexpr size_t count = 4 * 256 * 256;
    std::vector<Input> input(count);
    for (size_t i = 0; i < count; ++i) {
        const int lane = int(i >> 16), shift = 8 * lane;
        const uint32_t mask = ~(255u << shift);
        const uint32_t a = (0x80ff007fu & mask) | (uint32_t((i >> 8) & 255) << shift);
        const uint32_t b = (0x7f00ff80u & mask) | (uint32_t(i & 255) << shift);
        input[i] = {int32_t(a), int32_t(b), int32_t(i % 131071) - 65535,
                    float(int(i % 9) - 4) * 0.5f, float(int((i / 9) % 9) - 4) * 0.5f,
                    float(int((i / 81) % 9) - 4) * 0.5f};
    }
    Input* gpu_input = nullptr;
    Output* gpu_output = nullptr;
    CHECK_CUDA(cudaMalloc(&gpu_input, count * sizeof(Input)));
    CHECK_CUDA(cudaMalloc(&gpu_output, count * sizeof(Output)));
    CHECK_CUDA(cudaMemcpy(gpu_input, input.data(), count * sizeof(Input), cudaMemcpyHostToDevice));
    Check_Arithmetic<<<unsigned((count + 255) / 256), 256>>>(gpu_input, gpu_output, count);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<Output> output(count);
    CHECK_CUDA(cudaMemcpy(output.data(), gpu_output, count * sizeof(Output), cudaMemcpyDeviceToHost));
    size_t mismatches = 0;
    for (size_t i = 0; i < count; ++i) {
        const Input v = input[i];
        int32_t dot = v.c;
        for (int lane = 0; lane < 4; ++lane)
            dot += Signed_Byte(uint32_t(v.a), lane) * Signed_Byte(uint32_t(v.b), lane);
        // Half inputs and products are exactly representable, so no tolerance hides errors.
        const float half = v.x * v.y + v.z;
        const float fused = std::fma(v.x, v.y, v.z);
        const Output actual = output[i];
        if (actual.dot != dot || actual.half_lo != half || actual.half_hi != -half || actual.fused != fused)
            ++mismatches;
    }
    CHECK_CUDA(cudaFree(gpu_output));
    CHECK_CUDA(cudaFree(gpu_input));
    std::printf("cases=%zu mismatches=%zu (signed-byte dot, half2 arithmetic, float32 FMA)\n", count, mismatches);
    return mismatches == 0 ? 0 : 1;
}
