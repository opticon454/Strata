#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace strata::prefill::ws::detail::q20_half_lanes {

__device__ __forceinline__ __half2 Unpack_Half2(const uint32_t bits) {
    __half2_raw raw;
    raw.x = static_cast<unsigned short>(bits);
    raw.y = static_cast<unsigned short>(bits >> 16);
    return __half2(raw);
}

__device__ __forceinline__ uint32_t Pack_Half2(const __half2 value) {
    const __half2_raw raw = value;
    return uint32_t(raw.x) | (uint32_t(raw.y) << 16);
}

// The Q2 path uses integer half-lanes with exact sums within +/-2048.
__device__ __forceinline__ uint32_t hfma2(const uint32_t a, const uint32_t b, const uint32_t c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
    // Maxwell has no half arithmetic; these bounded integer operations are exact in FP32.
    const float2 x = __half22float2(Unpack_Half2(a));
    const float2 y = __half22float2(Unpack_Half2(b));
    const float2 z = __half22float2(Unpack_Half2(c));
    return Pack_Half2(__floats2half2_rn(fmaf(x.x, y.x, z.x), fmaf(x.y, y.y, z.y)));
#else
    uint32_t d;
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
#endif
}

__device__ __forceinline__ uint32_t hadd2(const uint32_t a, const uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
    const float2 x = __half22float2(Unpack_Half2(a));
    const float2 y = __half22float2(Unpack_Half2(b));
    return Pack_Half2(__floats2half2_rn(x.x + y.x, x.y + y.y));
#else
    uint32_t d;
    asm("add.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
#endif
}

}  // namespace strata::prefill::ws::detail::q20_half_lanes
