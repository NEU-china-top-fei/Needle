cmake_minimum_required(VERSION 3.24)
project(needle_gpu_project LANGUAGES CXX)

option(NEEDLE_CUDA "Build CUDA backend" ON)
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
                OUTPUT_VARIABLE PYBIND_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
                COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED PATHS "${PYBIND_DIR}" NO_DEFAULT_PATH)

# Keep compiled modules and the importable Python package out of the source tree.
set(SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
file(GLOB_RECURSE PYTHON_SOURCES CONFIGURE_DEPENDS RELATIVE "${SOURCE_ROOT}/python"
     "${SOURCE_ROOT}/python/*.py")
foreach(SOURCE IN LISTS PYTHON_SOURCES)
  configure_file("${SOURCE_ROOT}/python/${SOURCE}" "python/${SOURCE}" COPYONLY)
endforeach()
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/needle/backend_ndarray")
pybind11_add_module(ndarray_backend_cpu "${SOURCE_ROOT}/src/ndarray_backend_cpu.cc")
target_compile_features(ndarray_backend_cpu PRIVATE cxx_std_14)

if(NEEDLE_CUDA)
  # SM80 cubin plus compute_80 PTX supports the note's SM89 RTX 4060 via JIT.
  # Override with -DCMAKE_CUDA_ARCHITECTURES=89 for a native 4060 build.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
  endif()
  foreach(ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(NOT ARCH MATCHES "^[0-9]+(-real|-virtual)?$" OR ARCH LESS 80)
      message(FATAL_ERROR "Select explicit SM80+ CUDA architectures, e.g. 80 or 89")
    endif()
  endforeach()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  pybind11_add_module(ndarray_backend_cuda "${SOURCE_ROOT}/src/ndarray_backend_cuda.cu")
  target_link_libraries(ndarray_backend_cuda PRIVATE CUDA::cublas)
  set_target_properties(ndarray_backend_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()
