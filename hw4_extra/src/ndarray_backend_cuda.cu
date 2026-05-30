#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <cmath>

namespace needle
{
  namespace cuda
  {

#define BASE_THREAD_NUM 256

#define TILE 4
    typedef float scalar_t;
    const size_t ELEM_SIZE = sizeof(scalar_t);

    struct CudaArray
    {
      CudaArray(const size_t size)
      {
        cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
        if (err != cudaSuccess)
          throw std::runtime_error(cudaGetErrorString(err));
        this->size = size;
      }
      ~CudaArray() { cudaFree(ptr); }
      size_t ptr_as_int() { return (size_t)ptr; }

      scalar_t *ptr;
      size_t size;
    };

    struct CudaDims
    {
      dim3 block, grid;
    };

    CudaDims CudaOneDim(size_t size)
    {
      /**
       * Utility function to get cuda dimensions for 1D call
       */
      CudaDims dim;
      size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
      dim.block = dim3(BASE_THREAD_NUM, 1, 1);
      dim.grid = dim3(num_blocks, 1, 1);
      return dim;
    }

#define MAX_VEC_SIZE 8
    struct CudaVec
    {
      uint32_t size;
      int32_t data[MAX_VEC_SIZE];
    };

    CudaVec VecToCuda(const std::vector<int32_t> &x)
    {
      CudaVec shape;
      if (x.size() > MAX_VEC_SIZE)
        throw std::runtime_error("Exceeded CUDA supported max dimesions");
      shape.size = x.size();
      for (size_t i = 0; i < x.size(); i++)
      {
        shape.data[i] = x[i];
      }
      return shape;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Fill call
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void FillKernel(scalar_t *out, scalar_t val, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = val;
    }

    void Fill(CudaArray *out, scalar_t val)
    {
      CudaDims dim = CudaOneDim(out->size);
      FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Compact and setitem cals
    ////////////////////////////////////////////////////////////////////////////////

    // Untility function to convert contiguous index i to memory location from strides

    __device__ size_t idx2off(const CudaVec &strides, const CudaVec shape, size_t offset, size_t real)
    {
      size_t ret = offset;
      for (int32_t i = shape.size - 1; i >= 0; i--)
      {
        size_t temp = real % shape.data[i];
        ret += temp * strides.data[i];
        real = real / shape.data[i];
      }
      return ret;
    }

    __global__ void CompactKernel(const scalar_t *a, scalar_t *out, size_t size, CudaVec shape,
                                  CudaVec strides, size_t offset)
    {
      /**
       * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the
       * non-compact input a, to the corresponding item (at location gid) in the compact array out.
       *
       * Args:
       *   a: CUDA pointer to a array
       *   out: CUDA point to out array
       *   size: size of out array
       *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
       *   strides: vector of strides of out array
       *   offset: offset of out array
       */
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

      /// BEGIN SOLUTION
      if (gid < size)
      {
        size_t a_poi = idx2off(strides, shape, offset, gid);
        out[gid] = a[a_poi];
      }
      /// END SOLUTION
    }

    void Compact(const CudaArray &a, CudaArray *out, std::vector<int32_t> shape,
                 std::vector<int32_t> strides, size_t offset)
    {
      /**
       * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the
       * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give
       * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
       * the functions after this, however, you'll need to define these kernels as you see fit to
       * execute the underlying function.
       *
       * Args:
       *   a: non-compact represntation of the array, given as input
       *   out: compact version of the array to be written
       *   shape: shapes of each dimension for a and out
       *   strides: strides of the *a* array (not out, which has compact strides)
       *   offset: offset of the *a* array (not out, which has zero offset, being compact)
       */

      // Nothing needs to be added here
      CudaDims dim = CudaOneDim(out->size);
      CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                             VecToCuda(strides), offset);
    }

    __global__ void EwiseSetitemKernel(const scalar_t *a, scalar_t *out, CudaVec shape, size_t size,
                                       CudaVec strides, size_t offset)
    {
      size_t a_poi = blockDim.x * blockIdx.x + threadIdx.x;
      if (a_poi < size)
      {
        size_t out_poi = idx2off(strides, shape, offset, a_poi);
        out[out_poi] = a[a_poi];
      }
    }

    void EwiseSetitem(const CudaArray &a, CudaArray *out, std::vector<int32_t> shape,
                      std::vector<int32_t> strides, size_t offset)
    {
      /**
       * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
       * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
       *
       * Args:
       *   a: _compact_ array whose items will be written to out
       *   out: non-compact array whose items are to be written
       *   shape: shapes of each dimension for a and out
       *   strides: strides of the *out* array (not a, which has compact strides)
       *   offset: offset of the *out* array (not a, which has zero offset, being compact)
       */
      /// BEGIN SOLUTION
      CudaDims dim = CudaOneDim(a.size);
      EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, VecToCuda(shape), a.size,
                                                  VecToCuda(strides), offset);
      /// END SOLUTION
    }

    __global__ void ScalarSetitemKernel(scalar_t a, scalar_t *out, size_t size, CudaVec shape,
                                        CudaVec strides, size_t offset)
    {
      size_t out_poi = blockDim.x * blockIdx.x + threadIdx.x;
      if (out_poi < size)
      {
        size_t poi = idx2off(strides, shape, offset, out_poi);
        out[poi] = a;
      }
    }

    void ScalarSetitem(size_t size, scalar_t val, CudaArray *out, std::vector<int32_t> shape,
                       std::vector<int32_t> strides, size_t offset)
    {
      /**
       * Set items is a (non-compact) array
       *
       * Args:
       *   size: number of elements to write in out array (note that this will note be the same as
       *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
       *         product of items in shape, but covenient to just pass it here.
       *   val: scalar value to write to
       *   out: non-compact array whose items are to be written
       *   shape: shapes of each dimension of out
       *   strides: strides of the out array
       *   offset: offset of the out array
       */
      /// BEGIN SOLUTION
      CudaDims dim = CudaOneDim(size);
      ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, out->size, VecToCuda(shape),
                                                   VecToCuda(strides), offset);
      /// END SOLUTION
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Elementwise and scalar operations
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void EwiseAddKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
    {
      // Calculate the global index of the thread.
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = a[gid] + b[gid];
    }

    void EwiseAdd(const CudaArray &a, const CudaArray &b, CudaArray *out)
    {
      /**
       * Add together two CUDA arrays.
       * Args:
       *   a: Input array 'a' to be added
       *   b: Input array 'b' to be added
       *   out: Output array to store the result of 'a + b'
       */
      CudaDims dim = CudaOneDim(out->size);

      // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
      EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    __global__ void ScalarAddKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
    {
      // Calculate the global index of the thread.
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = a[gid] + val;
    }

    void ScalarAdd(const CudaArray &a, scalar_t val, CudaArray *out)
    {
      /**
       * Add a scalar value to every element of a CUDA array.
       * Args:
       *   a: Input array 'a'
       *   val: Scalar value to be added
       *   out: Output array to store the result of 'a + val'
       */
      CudaDims dim = CudaOneDim(out->size);

      // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a',
      // and store the result in array 'out'.
      ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

    /**
     * In the code the follows, use the above template to create analogous elementise
     * and and scalar operators for the following functions.  See the numpy backend for
     * examples of how they should work.
     *   - EwiseMul, ScalarMul
     *   - EwiseDiv, ScalarDiv
     *   - ScalarPower
     *   - EwiseMaximum, ScalarMaximum
     *   - EwiseEq, ScalarEq
     *   - EwiseGe, ScalarGe
     *   - EwiseLog
     *   - EwiseExp
     *   - EwiseTanh
     *
     * If you implement all these naively, there will be a lot of repeated code, so
     * you are welcome (but not required), to use macros or templates to define these
     * functions (however you want to do so, as long as the functions match the proper)
     * signatures above.
     */
    template <typename Func>
    __global__ void EwiseOpKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size, Func op)
    {
      size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
      if (idx < size)
      {
        out[idx] = op(a[idx], b[idx]);
      }
    }

    template <typename Func>
    void EwiseOp(const CudaArray &a, const CudaArray &b, CudaArray *out, Func op)
    {
      CudaDims dim = CudaOneDim(a.size);
      EwiseOpKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, a.size, op);
    }

    template <typename Func>
    __global__ void ScalarOpKernel(const scalar_t *a, const scalar_t val, scalar_t *out, size_t size, Func op)
    {
      size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
      if (idx < size)
      {
        out[idx] = op(a[idx], val);
      }
    }

    template <typename Func>
    void ScalarOp(const CudaArray &a, scalar_t val, CudaArray *out, Func op)
    {
      CudaDims dim = CudaOneDim(a.size);
      ScalarOpKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, a.size, op);
    }

    template <typename Func>
    __global__ void SinOpKernel(const scalar_t *a, scalar_t *out, size_t size, Func op)
    {
      size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
      if (idx < size)
      {
        out[idx] = op(a[idx]);
      }
    }
    template <typename Func>
    void SinOp(const CudaArray &a, CudaArray *out, Func op)
    {
      CudaDims dim = CudaOneDim(a.size);
      SinOpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, op);
    }

    struct MUL
    {
      __device__ scalar_t operator()(scalar_t a, scalar_t b) { return a * b; }
    };
    struct DIV
    {
      __device__ scalar_t operator()(scalar_t a, scalar_t b) { return a / b; }
    };
    struct POWER
    {
      __device__ scalar_t operator()(scalar_t a, scalar_t b) { return pow(a, b); }
    };
    struct MAX
    {
      __device__ scalar_t operator()(scalar_t a, scalar_t b) { return max(a, b); }
    };
    struct EQ
    {
      __device__ bool operator()(scalar_t a, scalar_t b) { return a == b; }
    };
    struct GE
    {
      __device__ scalar_t operator()(scalar_t a, scalar_t b) { return a >= b; }
    };
    struct LOG
    {
      __device__ scalar_t operator()(scalar_t a) { return log(a); }
    };
    struct EXP
    {
      __device__ scalar_t operator()(scalar_t a) { return exp(a); }
    };
    struct TANH
    {
      __device__ scalar_t operator()(scalar_t a) { return tanh(a); }
    };

#define REGISTERELE(NAME, FUNC) \
  m.def(NAME, [](const CudaArray &a, const CudaArray &b, CudaArray *out) { EwiseOp(a, b, out, FUNC); })

#define REGISTERSCA(NAME, FUNC) \
  m.def(NAME, [](const CudaArray &a, const scalar_t val, CudaArray *out) { ScalarOp(a, val, out, FUNC); })

#define REGISTERSIN(NAME, FUNC) \
  m.def(NAME, [](const CudaArray &a, CudaArray *out) { SinOp(a, out, FUNC); })

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////
#define TIEL 4
#define V 2
#define TILE 4
#define V 2

    __global__ void mm(const scalar_t *a, const scalar_t *b, scalar_t *c, size_t m, size_t n, size_t p)
    {
      // 1. 定义共享内存 (4x4)
      __shared__ scalar_t share_a[TILE][TILE];
      __shared__ scalar_t share_b[TILE][TILE];

      // 2. 寄存器私有变量
      scalar_t tempc[V][V];
      // 初始化寄存器
      for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
          tempc[i][j] = 0;

      size_t xblock = blockIdx.x, yblock = blockIdx.y;
      size_t xthread = threadIdx.x, ythread = threadIdx.y;

      // 线性线程 ID，用于协同搬运数据
      size_t tid = ythread * blockDim.x + xthread;

      // 外层循环：沿公共维 N 移动 TILE 步长
      for (size_t k_offset = 0; k_offset < n; k_offset += TILE)
      {

        // --- 阶段 1: 协同搬运数据到 Shared Memory ---
        // 整个 Block 有 256 个线程，但 TILE*TILE 只有 16 个元素
        // 我们只让前 16 个线程参与搬运
        if (tid < TILE * TILE)
        {
          size_t row = tid / TILE;
          size_t col = tid % TILE;

          // 搬运 A 的一个块 (M, N) -> (xblock*TILE + row, k_offset + col)
          if ((xblock * TILE + row) < m && (k_offset + col) < n)
            share_a[row][col] = a[(xblock * TILE + row) * n + (k_offset + col)];
          else
            share_a[row][col] = 0;

          // 搬运 B 的一个块 (N, P) -> (k_offset + row, yblock*TILE + col)
          if ((k_offset + row) < n && (yblock * TILE + col) < p)
            share_b[row][col] = b[(k_offset + row) * p + (yblock * TILE + col)];
          else
            share_b[row][col] = 0;
        }

        // 同步：确保所有线程都搬运完成了
        __syncthreads();

        // --- 阶段 2: 寄存器计算 (Register Tiling) ---
        // 只有前 (TILE/V) * (TILE/V) 个线程需要工作
        // 对于 TILE=4, V=2，只有 2x2=4 个线程在工作
        if (xthread < (TILE / V) && ythread < (TILE / V))
        {
          for (size_t k = 0; k < TILE; k++)
          {
            // 读取数据到寄存器并计算外积
            for (int i = 0; i < V; i++)
            {
              for (int j = 0; j < V; j++)
              {
                tempc[i][j] += share_a[xthread * V + i][k] * share_b[k][ythread * V + j];
              }
            }
          }
        }

        // 同步：等待计算完成，再进入下一轮搬运防止覆盖 shared memory
        __syncthreads();
      }

      // --- 阶段 3: 写回全局内存 ---
      if (xthread < (TILE / V) && ythread < (TILE / V))
      {
        for (int i = 0; i < V; i++)
        {
          for (int j = 0; j < V; j++)
          {
            size_t global_r = xblock * TILE + xthread * V + i;
            size_t global_c = yblock * TILE + ythread * V + j;
            if (global_r < m && global_c < p)
            {
              c[global_r * p + global_c] = tempc[i][j];
            }
          }
        }
      }
    }
    void Matmul(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M, uint32_t N,
                uint32_t P)
    {
      /**
       * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
       * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
       * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
       * over (i,j) entries in the output array.  However, to really get the full benefit of this
       * problem, we would encourage you to use cooperative fetching, shared memory register tiling,
       * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
       * the CPU backend, here you should implement a single function that works across all size
       * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
       * implementations, this function here will largely just set up the kernel call, and you should
       * implement the logic in a separate MatmulKernel() call.
       *
       *
       * Args:
       *   a: compact 2D array of size m x n
       *   b: comapct 2D array of size n x p
       *   out: compact 2D array of size m x p to write the output to
       *   M: rows of a / out
       *   N: columns of a / rows of b
       *   P: columns of b / out
       */

      /// BEGIN SOLUTION
      CudaDims dim;
      size_t x = (M - 1 + TILE) / TILE;
      size_t y = (P - 1 + TILE) / TILE;
      dim.grid = dim3(x, y, 1);
      dim.block = dim3(16, 16, 1);
      mm<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
      /// END SOLUTION
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Max and sum reductions
    ////////////////////////////////////////////////////////////////////////////////
    __global__ void ReduceMaxKernel(const scalar_t *a, scalar_t *out, size_t size, size_t reduce_size)
    {
      size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
      if (idx < size)
      {
        size_t begin = idx * reduce_size;
        out[idx] = a[begin];
        size_t end = (begin + reduce_size) > size * reduce_size ? size * reduce_size : begin + reduce_size;
        for (size_t i = begin; i < end; i++)
        {
          if (a[i] > out[idx])
            out[idx] = a[i];
        }
      }
    }

    void ReduceMax(const CudaArray &a, CudaArray *out, size_t reduce_size)
    {
      /**
       * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
       * for simplicity you can perform each reduction in a single CUDA thread.
       *
       * Args:
       *   a: compact array of size a.size = out.size * reduce_size to reduce over
       *   out: compact array to write into
       *   redice_size: size of the dimension to reduce over
       */
      /// BEGIN SOLUTION
      CudaDims dim = CudaOneDim(out->size);
      ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
      /// END SOLUTION
    }

    __global__ void ReduceSumKernel(const scalar_t *a, scalar_t *out, size_t size, size_t reduce_size)
    {
      size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
      if (idx < size)
      {
        size_t begin = idx * reduce_size;
        out[idx] = 0;
        size_t end = (begin + reduce_size) > size * reduce_size ? size * reduce_size : begin + reduce_size;
        for (size_t i = begin; i < end; i++)
        {
          out[idx] += a[i];
        }
      }
    }
    void ReduceSum(const CudaArray &a, CudaArray *out, size_t reduce_size)
    {
      /**
       * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you
       * can perform each reduction in a single CUDA thread.
       *
       * Args:
       *   a: compact array of size a.size = out.size * reduce_size to reduce over
       *   out: compact array to write into
       *   redice_size: size of the dimension to reduce over
       */
      /// BEGIN SOLUTION
      CudaDims dim = CudaOneDim(out->size);
      ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
      /// END SOLUTION
    }

    ////////////////////////////////////////////////////////////////////////////////
    // FlashAttention: fused scaled dot-product attention
    ////////////////////////////////////////////////////////////////////////////////

    // Process D in tiles to avoid register spilling for large head dims.
    // Each thread handles one Q row; online softmax avoids materializing NxN.
    __global__ void FlashAttnFwdKernel(
        const scalar_t *pQ, const scalar_t *pK, const scalar_t *pVal,
        scalar_t *pOut,
        uint32_t B, uint32_t H, uint32_t N, uint32_t D,
        bool causal, float softmax_scale)
    {
      size_t total_rows = B * H * N;
      size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx >= total_rows) return;

      // Decode linear idx → (b, h, query_pos)
      size_t i = idx % N;            // query position
      size_t h = (idx / N) % H;      // head index
      size_t b = idx / (N * H);      // batch index

      size_t head_off = (b * H + h) * N * D;
      const scalar_t *Qi = pQ + head_off + i * D;
      scalar_t       *Oi = pOut + head_off + i * D;

      // ── online softmax state ──
      float m_i = -1e30f;
      float l_i = 0.0f;

      // Zero the output (used as accumulator)
      for (size_t d = 0; d < D; d++) Oi[d] = 0.0f;

      // ── scan over all K/V positions ──
      for (size_t j = 0; j < N; j++)
      {
        if (causal && j > i) continue;

        const scalar_t *Kj = pK + head_off + j * D;

        // Dot product Qi · Kj
        float score = 0.0f;
        for (size_t d = 0; d < D; d++) score += Qi[d] * Kj[d];
        score *= softmax_scale;

        // Online softmax update
        float m_new = fmaxf(m_i, score);
        float exp_val = expf(score - m_new);
        float l_new  = l_i * expf(m_i - m_new) + exp_val;
        float rescale = expf(m_i - m_new);

        const scalar_t *Vj = pVal + head_off + j * D;
        for (size_t d = 0; d < D; d++)
          Oi[d] = Oi[d] * rescale + exp_val * Vj[d];

        m_i = m_new;
        l_i = l_new;
      }

      // ── normalize ──
      float inv_l = 1.0f / l_i;
      for (size_t d = 0; d < D; d++) Oi[d] *= inv_l;
    }

    void FlashAttention(
        const CudaArray &q, const CudaArray &k, const CudaArray &v,
        CudaArray *out,
        uint32_t B, uint32_t H, uint32_t N, uint32_t D,
        bool causal, float softmax_scale)
    {
      size_t total_rows = B * H * N;
      CudaDims dim = CudaOneDim(total_rows);
      FlashAttnFwdKernel<<<dim.grid, dim.block>>>(
          q.ptr, k.ptr, v.ptr, out->ptr,
          B, H, N, D, causal, softmax_scale);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Fused LayerNorm
    ////////////////////////////////////////////////////////////////////////////////

    // Block-wide reduction helper: sum across all threads in the block
    __device__ float block_reduce_sum(float val, float *smem, int block_size) {
      int tid = threadIdx.x;
      smem[tid] = val;
      __syncthreads();
      for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
      }
      return smem[0];
    }

    __global__ void LayerNormFwdKernel(
        const scalar_t *x, const scalar_t *weight, const scalar_t *bias,
        scalar_t *out,
        uint32_t D, float eps)
    {
      // One block per row; N = gridDim.x
      int row = blockIdx.x;
      int block_size = blockDim.x;
      int tid = threadIdx.x;

      const scalar_t *xi = x + row * D;
      scalar_t *oi = out + row * D;

      extern __shared__ float smem[];

      // ── pass 1: mean ──
      float local_sum = 0.0f;
      for (int d = tid; d < D; d += block_size)
        local_sum += xi[d];
      float mean = block_reduce_sum(local_sum, smem, block_size) / D;

      // ── pass 2: variance ──
      float local_var = 0.0f;
      for (int d = tid; d < D; d += block_size) {
        float diff = xi[d] - mean;
        local_var += diff * diff;
      }
      float var = block_reduce_sum(local_var, smem, block_size) / D;

      // ── pass 3: normalize + scale + shift ──
      float inv_std = rsqrtf(var + eps);
      for (int d = tid; d < D; d += block_size)
        oi[d] = weight[d] * (xi[d] - mean) * inv_std + bias[d];
    }

    void LayerNorm(
        const CudaArray &x, const CudaArray &weight, const CudaArray &bias,
        CudaArray *out,
        uint32_t N, uint32_t D, float eps)
    {
      dim3 grid(N);
      dim3 block(BASE_THREAD_NUM);
      size_t smem_bytes = BASE_THREAD_NUM * sizeof(float);
      LayerNormFwdKernel<<<grid, block, smem_bytes>>>(
          x.ptr, weight.ptr, bias.ptr, out->ptr, D, eps);
    }

  } // namespace cuda
} // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m)
{
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray &a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset)
        {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer); });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray *out)
        {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err)); });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);
  m.def("flash_attention",FlashAttention);
  m.def("layernorm",LayerNorm);

  REGISTERELE("ewise_mul", MUL());
  REGISTERSCA("scalar_mul", MUL());
  REGISTERELE("ewise_div", DIV());
  REGISTERSCA("scalar_div", DIV());
  REGISTERSCA("scalar_power", POWER());
  REGISTERELE("ewise_maximum", MAX());
  REGISTERSCA("scalar_maximum", MAX());
  REGISTERELE("ewise_eq", EQ());
  REGISTERSCA("scalar_eq", EQ());
  REGISTERELE("ewise_ge", GE());
  REGISTERSCA("scalar_ge", GE());
  REGISTERSIN("ewise_log", LOG());
  REGISTERSIN("ewise_exp", EXP());
  REGISTERSIN("ewise_tanh", TANH());

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
