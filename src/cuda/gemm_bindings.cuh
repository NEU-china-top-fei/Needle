// Included inside namespace needle::cuda, after CudaArray and Matmul.
inline void GemmCheck(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

inline void BlasCheck(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS error " + std::to_string(int(status)));
}

struct GemmBlas {
  cublasHandle_t handle;
  GemmBlas() { BlasCheck(cublasCreate(&handle)); }
  ~GemmBlas() { cublasDestroy(handle); }
  GemmBlas(const GemmBlas&) = delete;
};

inline bool TensorCoreAvailable() {
  int device;
  cudaDeviceProp prop;
  if (cudaGetDevice(&device) != cudaSuccess) { cudaGetLastError(); return false; }
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    cudaGetLastError(); return false;
  }
  return prop.major >= 8;
}

template<typename T> struct GemmBuffer {
  T* ptr = nullptr;
  explicit GemmBuffer(size_t count) { GemmCheck(cudaMalloc(&ptr, count * sizeof(T))); }
  ~GemmBuffer() { if (ptr) cudaFree(ptr); }
  GemmBuffer(const GemmBuffer&) = delete;
  GemmBuffer& operator=(const GemmBuffer&) = delete;
};

struct GemmEvent {
  cudaEvent_t event;
  GemmEvent() { GemmCheck(cudaEventCreate(&event)); }
  ~GemmEvent() { cudaEventDestroy(event); }
  GemmEvent(const GemmEvent&) = delete;
};

class PreparedGemm {
 public:
  int m, k, n, pm, pk, pn, device;
  std::unique_ptr<GemmBuffer<half>> a, b;
  std::unique_ptr<GemmBuffer<float>> c;
  std::unique_ptr<GemmBlas> blas;

  PreparedGemm(const CudaArray& lhs, const CudaArray& rhs, int M, int K, int N)
      : m(M), k(K), n(N) {
    if (m <= 0 || n <= 0 || k <= 0 || m > 64 * 65535 ||
        n > std::numeric_limits<int>::max() - 64 ||
        k > std::numeric_limits<int>::max() - 32)
      throw std::invalid_argument("Tensor Core GEMM requires positive, representable dimensions");
    if (lhs.size != size_t(m) * k || rhs.size != size_t(k) * n)
      throw std::invalid_argument("GEMM dimensions do not match compact input storage");
    if (!TensorCoreAvailable())
      throw std::runtime_error("Tensor Core GEMM requires a working SM80+ CUDA device");
    GemmCheck(cudaGetDevice(&device));
    pm = (m + 63) / 64 * 64; pk = (k + 31) / 32 * 32; pn = (n + 63) / 64 * 64;
    a.reset(new GemmBuffer<half>(size_t(pm) * pk));
    b.reset(new GemmBuffer<half>(size_t(pk) * pn));
    c.reset(new GemmBuffer<float>(size_t(pm) * pn));
    gemm::pack_half<<<256, 256>>>(lhs.ptr, a->ptr, m, k, pm, pk);
    GemmCheck(cudaGetLastError());
    gemm::pack_half<<<256, 256>>>(rhs.ptr, b->ptr, k, n, pk, pn);
    GemmCheck(cudaGetLastError());
  }

  void CheckDevice() const {
    int current; GemmCheck(cudaGetDevice(&current));
    if (current != device) throw std::runtime_error("PreparedGemm belongs to a different CUDA device");
  }

  void RunKernel(bool pipeline) {
    CheckDevice();
    dim3 grid(pn / 64, pm / 64);
    if (pipeline)
      gemm::tensorcore_kernel<true><<<grid, 128>>>(a->ptr, b->ptr, c->ptr, pn, pk);
    else
      gemm::tensorcore_kernel<false><<<grid, 128>>>(a->ptr, b->ptr, c->ptr, pn, pk);
    GemmCheck(cudaGetLastError());
  }

  void Run(CudaArray* out, bool pipeline) {
    if (out->size != size_t(m) * n) throw std::invalid_argument("GEMM output size mismatch");
    RunKernel(pipeline);
    gemm::crop_output<<<256, 256>>>(c->ptr, out->ptr, m, n, pn);
    GemmCheck(cudaGetLastError());
  }

  void RunBlasKernel() {
    CheckDevice();
    if (!blas) blas.reset(new GemmBlas());
    const float alpha = 1.f, beta = 0.f;
    // Row-major C=A@B maps to column-major C^T=B^T@A^T. Same padded
    // FP16 inputs and FP32 output as the custom kernels, no FP16 output cast.
    BlasCheck(cublasGemmEx(blas->handle, CUBLAS_OP_N, CUBLAS_OP_N, pn, pm, pk,
        &alpha, b->ptr, CUDA_R_16F, pn, a->ptr, CUDA_R_16F, pk,
        &beta, c->ptr, CUDA_R_32F, pn, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  void RunBlas(CudaArray* out) {
    if (out->size != size_t(m) * n) throw std::invalid_argument("GEMM output size mismatch");
    RunBlasKernel();
    gemm::crop_output<<<256, 256>>>(c->ptr, out->ptr, m, n, pn);
    GemmCheck(cudaGetLastError());
  }

  float BenchmarkBlas(int iterations, int warmup) {
    if (iterations <= 0 || warmup < 0) throw std::invalid_argument("Invalid benchmark counts");
    if (!blas) blas.reset(new GemmBlas());
    for (int i = 0; i < warmup; ++i) RunBlasKernel();
    GemmCheck(cudaDeviceSynchronize());
    GemmEvent start, stop;
    GemmCheck(cudaEventRecord(start.event));
    for (int i = 0; i < iterations; ++i) RunBlasKernel();
    GemmCheck(cudaEventRecord(stop.event));
    GemmCheck(cudaEventSynchronize(stop.event));
    float ms; GemmCheck(cudaEventElapsedTime(&ms, start.event, stop.event));
    return ms / iterations;
  }

  // CUDA-event device time for the padded GEMM kernel only. Packing, allocation,
  // output cropping and Python dispatch are deliberately outside this interval.
  float Benchmark(int iterations, int warmup, bool pipeline) {
    if (iterations <= 0 || warmup < 0) throw std::invalid_argument("Invalid benchmark counts");
    for (int i = 0; i < warmup; ++i) RunKernel(pipeline);
    GemmCheck(cudaDeviceSynchronize());
    GemmEvent start, stop;
    GemmCheck(cudaEventRecord(start.event));
    for (int i = 0; i < iterations; ++i) RunKernel(pipeline);
    GemmCheck(cudaEventRecord(stop.event));
    GemmCheck(cudaEventSynchronize(stop.event));
    float ms; GemmCheck(cudaEventElapsedTime(&ms, start.event, stop.event));
    return ms / iterations;
  }
};

inline void TensorCoreMatmul(const CudaArray& a, const CudaArray& b, CudaArray* out,
                             int m, int k, int n, bool pipeline) {
  PreparedGemm packed(a, b, m, k, n);
  packed.Run(out, pipeline);
  // Temporary cudaFree calls may synchronize. Framework benchmarks include this
  // cost; the separate prepared-kernel benchmark does not.
}

inline void RegisterGemm(pybind11::module_& m) {
  namespace py = pybind11;
  m.def("tensorcore_available", TensorCoreAvailable);
  m.def("synchronize", [] { GemmCheck(cudaDeviceSynchronize()); });
  m.def("tensorcore_matmul", TensorCoreMatmul, py::arg("a"), py::arg("b"),
        py::arg("out"), py::arg("m"), py::arg("k"), py::arg("n"), py::arg("pipeline") = true);
  m.def("gemm_device_info", [] {
    int device; cudaDeviceProp prop;
    GemmCheck(cudaGetDevice(&device)); GemmCheck(cudaGetDeviceProperties(&prop, device));
    int runtime, driver; GemmCheck(cudaRuntimeGetVersion(&runtime));
    GemmCheck(cudaDriverGetVersion(&driver));
    py::dict d; d["name"] = prop.name; d["sm_major"] = prop.major;
    d["sm_minor"] = prop.minor; d["runtime"] = runtime; d["driver"] = driver;
    int version; GemmBlas handle; BlasCheck(cublasGetVersion(handle.handle, &version));
    d["cublas"] = version;
    return d;
  });
  py::class_<PreparedGemm>(m, "PreparedGemm")
      .def(py::init<const CudaArray&, const CudaArray&, int, int, int>())
      .def("run", &PreparedGemm::Run, py::arg("out"), py::arg("pipeline") = true)
      .def("run_cublas", &PreparedGemm::RunBlas)
      .def("benchmark_cublas", &PreparedGemm::BenchmarkBlas, py::arg("iterations") = 100,
           py::arg("warmup") = 10)
      .def("benchmark", &PreparedGemm::Benchmark, py::arg("iterations") = 100,
           py::arg("warmup") = 10, py::arg("pipeline") = true);
}
