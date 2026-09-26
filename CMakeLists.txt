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
    src/cpp/lsm/landscape.cpp
    src/cpp/lsm/metrics.cpp
)
target_include_directories(lsmcore PUBLIC src/cpp)
set_target_properties(lsmcore PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(gtbcore PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(LANDSCHAFT_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(gtbcore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(lsmcore PUBLIC OpenMP::OpenMP_CXX)
    else()
        message(WARNING "OpenMP not found: building single-threaded")
    endif()
endif()

# Results must match the original C code bit for bit: keep IEEE semantics
# (no fast-math, no FMA contraction).
if(MSVC)
    target_compile_options(gtbcore PRIVATE /fp:precise /W4 /permissive-)
    target_compile_options(lsmcore PRIVATE /fp:precise /W4 /permissive-)
else()
    target_compile_options(gtbcore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    target_compile_options(lsmcore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
endif()

pybind11_add_module(_core MODULE src/cpp/bindings.cpp)
target_link_libraries(_core PRIVATE gtbcore lsmcore)
if(MSVC)
    target_compile_options(_core PRIVATE /fp:precise)
else()
    target_compile_options(_core PRIVATE -ffp-contract=off -fno-fast-math)
endif()

install(TARGETS _core DESTINATION landschaft)
