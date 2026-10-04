#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}

size_t idx2off(const std::vector<int32_t>& index,const std::vector<int32_t>& strides,size_t offset){
  size_t temp=offset;
  int l=strides.size();
  for(int i=0;i<l;i++){
    temp+=index[i]*strides[i];
  }
  return temp;
}

bool incre(std::vector<int32_t>& index,const std::vector<int32_t>& shape){
  if(index.size()==0)return false;
  int l=shape.size();
  index[l-1]++;
  for(int i=l-1;i>=0;i--){
    if(index[i]>=shape[i]){
      if(i==0)return false;
      index[i]=0;
      index[i-1]++;
    }
  }
  return true;
}

void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
  int l=shape.size();
  std::vector<int32_t> idx(l,0);
  std::vector<int32_t> out_stride(l,1);
  for(int i=l-2;i>=0;i--){
    out_stride[i]=out_stride[i+1]*shape[i+1];
  }
  while(true){
    size_t aptr=idx2off(idx,strides,offset);
    size_t outptr=idx2off(idx,out_stride,0);
    out->ptr[outptr]=a.ptr[aptr];
    if(!incre(idx,shape))return;
  }
  /// END SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  int l=shape.size();
  std::vector<int32_t> idx(l,0);
  std::vector<int32_t> compstride(l,1);
  for(int i=l-2;i>=0;i--){
    compstride[i]=compstride[i+1]*shape[i+1];
  }
  while(true){
    size_t aptr=idx2off(idx,compstride,0);
    size_t outptr=idx2off(idx,strides,offset);
    out->ptr[outptr]=a.ptr[aptr];
    if(!incre(idx,shape))return;
  }
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  int l=shape.size();
  std::vector<int32_t> idx(l,0);
  while(true){
    size_t off=idx2off(idx,strides,offset);
    out->ptr[off]=val;
    if(!incre(idx,shape))return;
  }
  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
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



void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  memset(out->ptr,0,sizeof(scalar_t)*m*p);
  for(int i=0;i<m;i++){
    for(int j=0;j<n;j++){
      for(int k=0;k<p;k++){
        out->ptr[i*p+k]+=a.ptr[i*n+j]*b.ptr[j*p+k];
      }
    }
  }
  /// END SOLUTION
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for(int i=0;i<TILE;i++){
    for(int j=0;j<TILE;j++){
      for(int k=0;k<TILE;k++){
        out[i*TILE+k]+=a[i*TILE+j]*b[j*TILE+k];
      }
    }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  memset(out->ptr,0,sizeof(scalar_t)*m*p);
  uint32_t m_div=m/TILE,n_div=n/TILE,p_div=p/TILE;
  uint32_t a_stride0=TILE*TILE*n_div,a_stride1=TILE*TILE;
  uint32_t b_stride0=TILE*TILE*p_div,b_stride1=a_stride1;
  uint32_t c_stride0=b_stride0,c_stride1=a_stride1;
  for(uint32_t i=0;i<m_div;i++){
    for(uint32_t j=0;j<n_div;j++){
      float *atemp=a.ptr+(i*a_stride0+j*a_stride1);
      for(uint32_t k=0;k<p_div;k++){
        float *btemp=b.ptr+(j*b_stride0+k*b_stride1);
        float *outtemp=out->ptr+(i*c_stride0+k*c_stride1);
        AlignedDot(atemp,btemp,outtemp);
      }
    }
  }
  /// END SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  size_t s=(a.size-1+reduce_size)/reduce_size;
  for(size_t i=0;i<s;i++){
    size_t begin=i*reduce_size;
    size_t end=begin+reduce_size>a.size?a.size:begin+reduce_size;
    out->ptr[i]=a.ptr[i*reduce_size];
    for(size_t j=begin;j<end;j++){
      out->ptr[i]=std::max(out->ptr[i],a.ptr[j]);
    }
  }
  /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  size_t s=(a.size-1+reduce_size)/reduce_size;
  for(size_t i=0;i<s;i++){
    size_t begin=i*reduce_size;
    size_t end=begin+reduce_size>a.size?a.size:begin+reduce_size;
    out->ptr[i]=0;
    for(size_t j=begin;j<end;j++){
      out->ptr[i]+=a.ptr[j];
    }
  }
  /// END SOLUTION
}

void EwiseOp(const AlignedArray& a,const AlignedArray& b,AlignedArray* out,std::function<scalar_t(scalar_t,scalar_t)>op){
  for(size_t i=0;i<a.size;i++){
    out->ptr[i]=op(a.ptr[i],b.ptr[i]);
  }
}
void ScalarOp(const AlignedArray& a,scalar_t val,AlignedArray* out,std::function<scalar_t(scalar_t,scalar_t)>op){
  for(size_t i=0 ;i<a.size;i++){
    out->ptr[i]=op(a.ptr[i],val);
  }
}

#define REGISTELE(NAME,FUNC) m.def(NAME,[](const AlignedArray& a,const AlignedArray& b,AlignedArray* out){\
  EwiseOp(a,b,out,FUNC);\
})
#define REGISTSCA(NAME,FUNC) m.def(NAME,[](const AlignedArray& a,scalar_t val,AlignedArray* out){\
  ScalarOp(a,val,out,FUNC);\
})

#define REGISTSIN(NAME,FUNC) m.def(NAME,[](const AlignedArray& a,AlignedArray* out){\
  for(size_t i=0;i<a.size;i++){\
    out->ptr[i]=FUNC(a.ptr[i]);\
  }\
})

}  // namespace cpu
}  // namespace needle


PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  REGISTELE("ewise_mul", std::multiplies<scalar_t>());
  REGISTSCA("scalar_mul", std::multiplies<scalar_t>());
  REGISTELE("ewise_div", std::divides<scalar_t>());
  REGISTSCA("scalar_div", std::divides<scalar_t>());
  REGISTSCA("scalar_power", static_cast<scalar_t(*)(scalar_t, scalar_t)>(std::pow));
  REGISTELE("ewise_maximum", static_cast<scalar_t(*)(scalar_t, scalar_t)>(std::fmax));
  REGISTSCA("scalar_maximum", static_cast<scalar_t(*)(scalar_t, scalar_t)>(std::fmax));
  REGISTELE("ewise_eq", std::equal_to<scalar_t>());
  REGISTSCA("scalar_eq", std::equal_to<scalar_t>());
  REGISTELE("ewise_ge", std::greater_equal<scalar_t>());
  REGISTSCA("scalar_ge", std::greater_equal<scalar_t>());
  REGISTSIN("ewise_log", std::log);
  REGISTSIN("ewise_exp", std::exp);
  REGISTSIN("ewise_tanh", std::tanh);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}