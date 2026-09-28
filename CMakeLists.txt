cmake_minimum_required(VERSION 3.20)
project(landschaft LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LANDSCHAFT_OPENMP "Build with OpenMP" ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

add_library(gtbcore STATIC
    src/cpp/gtb/primitives.cpp
    src/cpp/gtb/thinning.cpp
    src/cpp/gtb/mspa.cpp
    src/cpp/gtb/gtn.cpp
    src/cpp/gtb/fad.cpp
    src/cpp/gtb/distance.cpp
    src/cpp/gtb/entropy.cpp
    src/cpp/gtb/contagion.cpp
    src/cpp/gtb/mosaic.cpp
    src/cpp/gtb/restoration.cpp
    src/cpp/gtb/morphology.cpp
    src/cpp/gtb/change.cpp
    src/cpp/gtb/grayspatcon.cpp
    src/cpp/gtb/accounting.cpp
    src/cpp/gtb/influence.cpp
    src/cpp/gtb/isochrone.cpp
    src/cpp/gtb/spa.cpp
    src/cpp/gtb/fos.cpp
)
target_include_directories(gtbcore PUBLIC src/cpp)

add_library(lsmcore STATIC
    src/cpp/lsm/geometry.cpp
    src/cpp/lsm/landscape.cpp
    src/cpp/lsm/metrics.cpp
    src/cpp/lsm/table.cpp
    src/cpp/lsm/tiled.cpp
    src/cpp/lsm/utils.cpp
    src/cpp/lsm/fragstats.cpp
    src/cpp/lsm/rasterize.cpp
    src/cpp/lsm/rlm.cpp
)
target_include_directories(lsmcore PUBLIC src/cpp)

# Eigen 3.4.0 (headers only) for the sparse solvers of the circuit module.
# FETCHCONTENT_SOURCE_DIR_EIGEN=<path> uses a local copy (offline builds).
include(FetchContent)
FetchContent_Declare(eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    # No CMakeLists.txt there: only download, do not configure Eigen's own project.
    SOURCE_SUBDIR headers-only)
FetchContent_MakeAvailable(eigen)
add_library(eigen_headers INTERFACE)
target_include_directories(eigen_headers SYSTEM INTERFACE ${eigen_SOURCE_DIR})
# Eigen's own threads stay off: the circuit kernels split their work in fixed
# blocks themselves, so results do not depend on the number of threads.
target_compile_definitions(eigen_headers INTERFACE EIGEN_MPL2_ONLY EIGEN_DONT_PARALLELIZE)

set(CIRCUIT_SOURCES
    src/cpp/circuit/circuitscape.cpp
    src/cpp/circuit/grid.cpp
    src/cpp/circuit/omniscape.cpp
    src/cpp/circuit/system.cpp
)
add_library(circuitcore STATIC ${CIRCUIT_SOURCES})
target_include_directories(circuitcore PUBLIC src/cpp)
target_link_libraries(circuitcore PRIVATE eigen_headers)
set_target_properties(circuitcore PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(conncore STATIC
    src/cpp/conn/cost.cpp
    src/cpp/conn/graph.cpp
    src/cpp/conn/nodes.cpp
)
target_include_directories(conncore PUBLIC src/cpp)
# R's x87 sqrt and 80-bit sums (lsm/rmath.hpp, lsm/extended.hpp).
target_link_libraries(conncore PUBLIC lsmcore)
set_target_properties(conncore PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Neutral landscape models (NLMR): R's and libstdc++'s generators, the models.
add_library(nlmcore STATIC
    src/cpp/nlm/rng.cpp
)
target_include_directories(nlmcore PUBLIC src/cpp)
# R's x87 log (lsm/rmath.hpp).
target_link_libraries(nlmcore PUBLIC lsmcore)
set_target_properties(nlmcore PROPERTIES POSITION_INDEPENDENT_CODE ON)
# R's x87 log/sqrt and LDOUBLE sums (see src/cpp/lsm/rmath.hpp): MSVC has no
# x64 inline assembly. x86-64 only; every other target uses the C++ code.
if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|x86_64)$")
    enable_language(ASM_MASM)
    target_sources(lsmcore PRIVATE src/cpp/lsm/x87_msvc.asm)
endif()
set_target_properties(lsmcore PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(gtbcore PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(LANDSCHAFT_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(gtbcore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(lsmcore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(conncore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(nlmcore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(circuitcore PUBLIC OpenMP::OpenMP_CXX)
    else()
        message(WARNING "OpenMP not found: building single-threaded")
    endif()
endif()

# Results must match the original C code bit for bit: keep IEEE semantics
# (no fast-math, no FMA contraction).
if(MSVC)
    target_compile_options(gtbcore PRIVATE /fp:precise /W4 /permissive-)
    target_compile_options(lsmcore PRIVATE /fp:precise /W4 /permissive-)
    target_compile_options(conncore PRIVATE /fp:precise /W4 /permissive-)
    target_compile_options(nlmcore PRIVATE /fp:precise /W4 /permissive-)
    target_compile_options(circuitcore PRIVATE /fp:precise /W4 /permissive-)
else()
    target_compile_options(gtbcore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    target_compile_options(lsmcore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    target_compile_options(conncore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    target_compile_options(nlmcore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    target_compile_options(circuitcore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
endif()

pybind11_add_module(_core MODULE src/cpp/bindings.cpp src/cpp/circuit/python.cpp src/cpp/nlm/python.cpp)
target_link_libraries(_core PRIVATE eigen_headers)
target_link_libraries(_core PRIVATE gtbcore lsmcore conncore circuitcore nlmcore)
if(MSVC)
    target_compile_options(_core PRIVATE /fp:precise)
else()
    target_compile_options(_core PRIVATE -ffp-contract=off -fno-fast-math)
endif()

install(TARGETS _core DESTINATION landschaft)

# The circuit module again, built for AVX2 and FMA, as a module of its own
# (landschaft._circuit_avx2): Python loads it only when the CPU has them
# (_core.cpu_supports_avx2_fma), which makes the dense kernels of the
# factorisations 1.2-1.5x faster. Same IEEE rules for the scalar code.
option(LANDSCHAFT_CIRCUIT_AVX2 "Build the AVX2 circuit module on x86-64" ON)
if(LANDSCHAFT_CIRCUIT_AVX2 AND CMAKE_SIZEOF_VOID_P EQUAL 8
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64)$" AND NOT CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    pybind11_add_module(_circuit_avx2 MODULE src/cpp/circuit/python.cpp ${CIRCUIT_SOURCES})
    target_include_directories(_circuit_avx2 PRIVATE src/cpp)
    target_link_libraries(_circuit_avx2 PRIVATE eigen_headers)
    target_compile_definitions(_circuit_avx2 PRIVATE LANDSCHAFT_CIRCUIT_AVX2_MODULE)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(_circuit_avx2 PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(MSVC)
        target_compile_options(_circuit_avx2 PRIVATE /arch:AVX2 /fp:precise)
    else()
        target_compile_options(_circuit_avx2 PRIVATE -mavx2 -mfma -ffp-contract=off -fno-fast-math)
    endif()
    install(TARGETS _circuit_avx2 DESTINATION landschaft)
endif()
