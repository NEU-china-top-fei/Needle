#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

// Adapted from the author's GEMM notes (64x64x32, four warps):
// https://neu-china-top-fei.github.io/2026/06/04/DLsysSet/GEMM/
// Inputs are padded FP16; accumulation/output are FP32. This header does not
// change Needle's FP32 storage contract. SM80+ is required by the host wrapper.
namespace needle { namespace gemm {
namespace wmma = nvcuda::wmma;
constexpr int BM = 64, BN = 64, BK = 32, THREADS = 128;

__device__ __forceinline__ void copy_async_16(half* dst, const half* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::
               "r"(address), "l"(src) : "memory");
#else
  *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
#endif
}

__device__ __forceinline__ void commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
__device__ __forceinline__ void wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

template<bool Async>
__device__ void load_tile(const half* a, const half* b, half* sa, half* sb,
                         int n, int k, int row, int col, int offset) {
  for (int chunk = threadIdx.x; chunk < 512; chunk += THREADS) {
    const int elem = (chunk % 256) * 8;
    half* dst;
    const half* src;
    if (chunk < 256) {
      dst = sa + elem;
      src = a + size_t(row + elem / BK) * k + offset + elem % BK;
    } else {
      dst = sb + elem;
      src = b + size_t(offset + elem / BN) * n + col + elem % BN;
    }
    if (Async) copy_async_16(dst, src);
    else *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
  }
}

template<bool Pipeline>
__global__ void tensorcore_kernel(const half* a, const half* b, float* c,
                                 int n, int k) {
  const int row = blockIdx.y * BM, col = blockIdx.x * BN;
  const int warp = threadIdx.x / 32;
  const int wr = (warp / 2) * 32, wc = (warp % 2) * 32;
  // WMMA load/store pointers require 256-bit alignment (stronger than cp.async).
  __shared__ __align__(32) half sa[2][BM][BK];
  __shared__ __align__(32) half sb[2][BK][BN];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  int stage = 0;
  if (Pipeline) {
    load_tile<true>(a, b, &sa[0][0][0], &sb[0][0][0], n, k, row, col, 0);
    commit(); wait(); __syncthreads();
  }
  for (int offset = 0; offset < k; offset += BK) {
    if (Pipeline) {
      if (offset + BK < k) {
        load_tile<true>(a, b, &sa[stage ^ 1][0][0], &sb[stage ^ 1][0][0],
                        n, k, row, col, offset + BK);
        commit();
      }
    } else {
      load_tile<false>(a, b, &sa[stage][0][0], &sb[stage][0][0],
                       n, k, row, col, offset);
      __syncthreads();
    }
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
      for (int i = 0; i < 2; ++i)
        wmma::load_matrix_sync(af[i], &sa[stage][wr + i * 16][kk], BK);
      for (int j = 0; j < 2; ++j)
        wmma::load_matrix_sync(bf[j], &sb[stage][kk][wc + j * 16], BN);
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
          wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    if (Pipeline) wait();
    // All warps must finish consuming a stage before it can be overwritten.
    __syncthreads();
    if (Pipeline) stage ^= 1;
  }
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      wmma::store_matrix_sync(c + size_t(row + wr + i * 16) * n + col + wc + j * 16,
                              acc[i][j], n, wmma::mem_row_major);
}

__global__ void pack_half(const float* in, half* out, int rows, int cols,
                          int padded_rows, int padded_cols) {
  const size_t total = size_t(padded_rows) * padded_cols;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += size_t(blockDim.x) * gridDim.x) {
    const size_t r = i / padded_cols, c = i % padded_cols;
    out[i] = __float2half_rn(r < rows && c < cols ? in[r * cols + c] : 0.f);
  }
}

__global__ void crop_output(const float* in, float* out, int m, int n, int pn) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size_t(m) * n;
       i += size_t(blockDim.x) * gridDim.x)
    out[i] = in[(i / n) * pn + i % n];
}
}}  // namespace needle::gemm
