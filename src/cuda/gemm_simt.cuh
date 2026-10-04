// Shared FP32 implementation extracted from hw4; guarded fallback for edge sizes.
// 优化GEMM,不检查尺寸对齐
#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define FETCH4(val) (reinterpret_cast<float4 *>(&(val))[0])
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_N = 128;
    const int BLOCK_SIZE_K = 8;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    const int THREAD_NUM_X = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_NUM_Y = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_PER_BLOCK = THREAD_NUM_X * THREAD_NUM_Y;
    const int load_global_to_reg_a = BLOCK_SIZE_M * BLOCK_SIZE_K / THREAD_PER_BLOCK;
    const int load_global_to_reg_b = BLOCK_SIZE_K * BLOCK_SIZE_N / THREAD_PER_BLOCK;
    const int THREAD_NUM_PER_A_ROW = BLOCK_SIZE_K / 4;
    const int THREAD_NUM_PER_B_ROW = BLOCK_SIZE_N / 4;
    const int STRIDE_A_TILE = THREAD_PER_BLOCK / THREAD_NUM_PER_A_ROW;
    const int STRIDE_B_TILE = THREAD_PER_BLOCK / THREAD_NUM_PER_B_ROW;
    __global__ void optkernel(const scalar_t *a, const scalar_t *b, scalar_t *c, uint32_t M, uint32_t N, uint32_t K)
    {
      // thread/block id
      int bx = blockIdx.x, by = blockIdx.y;
      int tx = threadIdx.x, ty = threadIdx.y;
      int tid = ty * THREAD_NUM_X + tx;

      // shared memory and register
      __shared__ scalar_t ashared[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
      __shared__ scalar_t bshared[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
      scalar_t aregister[2][THREAD_SIZE_Y];
      scalar_t bregitser[2][THREAD_SIZE_X];
      scalar_t cregister[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
      scalar_t global_to_reg_a[load_global_to_reg_a];
      scalar_t global_to_reg_b[load_global_to_reg_b];

      // location
      int A_ROW_START_INSIDE = tid / THREAD_NUM_PER_A_ROW;
      int B_ROW_START_INSIDE = tid / THREAD_NUM_PER_B_ROW;
      int A_COL_INSIDE = tid % THREAD_NUM_PER_A_ROW * 4;
      int B_COL_INSIDE = tid % THREAD_NUM_PER_B_ROW * 4;
      scalar_t *A = const_cast<scalar_t *>(a) + OFFSET(by * BLOCK_SIZE_M, 0, K);
      scalar_t *B = const_cast<scalar_t *>(b) + OFFSET(0, bx * BLOCK_SIZE_N, N);

      // prefetch
      // global to shared
      for (int i = 0; i < BLOCK_SIZE_M; i += STRIDE_A_TILE)
      {
        int reg_idx_a = i / (STRIDE_A_TILE / 4);
        FETCH4(global_to_reg_a[reg_idx_a]) = FETCH4(A[OFFSET(i + A_ROW_START_INSIDE, A_COL_INSIDE, K)]);
        for (int j = 0; j < 4; j++)
        {
          ashared[0][A_COL_INSIDE + j][i + A_ROW_START_INSIDE] = global_to_reg_a[reg_idx_a + j];
        }
      }
      for (int i = 0; i < BLOCK_SIZE_K; i += STRIDE_B_TILE)
      {
        FETCH4(bshared[0][i + B_ROW_START_INSIDE][B_COL_INSIDE]) = FETCH4(B[OFFSET(i + B_ROW_START_INSIDE, B_COL_INSIDE, N)]);
      }
      __syncthreads();
      // shared to register
      int smem_read = 0;
      int smem_write = 1;
      int write_tag = 0;
      for (int i = 0; i < THREAD_SIZE_Y; i += 4)
      {
        FETCH4(aregister[write_tag][i]) = FETCH4(ashared[smem_read][0][THREAD_SIZE_Y * ty + i]);
      }
      for (int i = 0; i < THREAD_SIZE_X; i += 4)
      {
        FETCH4(bregitser[write_tag][i]) = FETCH4(bshared[smem_read][0][THREAD_SIZE_X * tx + i]);
      }
      int tile_idx = 0;
      int read_tag = 0;
      write_tag = 1;
      do
      {
        tile_idx += BLOCK_SIZE_K;
        if (tile_idx < K)
        {
          for (int i = 0; i < BLOCK_SIZE_M; i += STRIDE_A_TILE)
          {
            int reg_idx_a = i / (STRIDE_A_TILE / 4);
            FETCH4(global_to_reg_a[reg_idx_a]) = FETCH4(A[OFFSET(i + A_ROW_START_INSIDE, A_COL_INSIDE + tile_idx, K)]);
          }
          for (int i = 0; i < BLOCK_SIZE_K; i += STRIDE_B_TILE)
          {
            int reg_idx_b = i / (STRIDE_B_TILE / 4);
            FETCH4(global_to_reg_b[reg_idx_b]) = FETCH4(B[OFFSET(i + B_ROW_START_INSIDE + tile_idx, B_COL_INSIDE, N)]);
          }
        }
        for (int k = 0; k < BLOCK_SIZE_K - 1; k++)
        {
          for (int i = 0; i < THREAD_SIZE_Y; i += 4)
          {
            FETCH4(aregister[write_tag][i]) = FETCH4(ashared[smem_read][k + 1][THREAD_SIZE_Y * ty + i]);
          }
          for (int i = 0; i < THREAD_SIZE_X; i += 4)
          {
            FETCH4(bregitser[write_tag][i]) = FETCH4(bshared[smem_read][k + 1][THREAD_SIZE_X * tx + i]);
          }
          for (int i = 0; i < THREAD_SIZE_X; i++)
          {
            for (int j = 0; j < THREAD_SIZE_Y; j++)
            {
              cregister[i][j] += aregister[read_tag][i] * bregitser[read_tag][j];
            }
          }
          int tmp_tag = write_tag;
          write_tag = read_tag;
          read_tag = tmp_tag;
        }

        if (tile_idx < K)
        {
          for (int i = 0; i < BLOCK_SIZE_M; i += STRIDE_A_TILE)
          {
            int reg_idx_a = i / (STRIDE_A_TILE / 4);
            for (int j = 0; j < 4; j++)
            {
              ashared[smem_write][A_COL_INSIDE + j][i + A_ROW_START_INSIDE] = global_to_reg_a[reg_idx_a + j];
            }
          }
          for (int i = 0; i < BLOCK_SIZE_K; i += STRIDE_B_TILE)
          {
            int reg_idx_b = i / (STRIDE_B_TILE / 4);
            FETCH4(bshared[smem_write][i + B_ROW_START_INSIDE][B_COL_INSIDE]) = FETCH4(global_to_reg_b[reg_idx_b]);
          }
          __syncthreads();
          smem_read = smem_read ^ smem_write;
          smem_write = smem_write ^ smem_read;
          smem_read = smem_read ^ smem_write;
        }
        for (int i = 0; i < THREAD_SIZE_Y; i += 4)
        {
          FETCH4(aregister[write_tag][i]) = FETCH4(ashared[smem_read][0][THREAD_SIZE_Y * ty + i]);
        }
        for (int i = 0; i < THREAD_SIZE_X; i += 4)
        {
          FETCH4(bregitser[write_tag][i]) = FETCH4(bshared[smem_read][0][THREAD_SIZE_X * tx + i]);
        }
        for (int i = 0; i < THREAD_SIZE_X; i++)
        {
          for (int j = 0; j < THREAD_SIZE_Y; j++)
          {
            cregister[i][j] += aregister[read_tag][i] * bregitser[read_tag][j];
          }
        }
        int tmp_tag = write_tag;
        write_tag = read_tag;
        read_tag = tmp_tag;
      } while (tile_idx < K);
      for (int i = 0; i < THREAD_SIZE_Y; i++)
      {
        for (int j = 0; j < THREAD_SIZE_X; j++)
        {
          int row = by * BLOCK_SIZE_M + THREAD_SIZE_Y * ty + i;
          int col = bx * BLOCK_SIZE_N + THREAD_SIZE_X * tx + j;
          c[OFFSET(row, col, N)] = cregister[i][j];
        }
      }
    }
    void OpMatmul(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M, uint32_t N,
                  uint32_t P)
    {
      if (!M || !N || !P) throw std::invalid_argument("GEMM requires positive dimensions");
      if (a.size != size_t(M) * N || b.size != size_t(N) * P || out->size != size_t(M) * P)
        throw std::invalid_argument("GEMM storage size mismatch");
      if (M % BLOCK_SIZE_M || P % BLOCK_SIZE_N || N % BLOCK_SIZE_K) {
        Matmul(a, b, out, M, N, P);
        GemmCheck(cudaGetLastError());
        return;
      }
      CudaDims d;
      d.grid = dim3(P / BLOCK_SIZE_N, M / BLOCK_SIZE_M, 1);
      d.block = dim3(BLOCK_SIZE_M / THREAD_SIZE_Y, BLOCK_SIZE_N / THREAD_SIZE_X, 1);
      optkernel<<<d.grid, d.block>>>(a.ptr, b.ptr, out->ptr, M, P, N);
      GemmCheck(cudaGetLastError());
    }

