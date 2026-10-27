cmake_minimum_required(VERSION 3.20)
project(landschaft LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LANDSCHAFT_OPENMP "Build with OpenMP" ON)
# OFF: only the static C++ libraries (and their headers) are built and installed, without Python, for
# native programs that link them (Terraflux); the MILP prioritization (HiGHS) only with LANDSCHAFT_MILP.
option(LANDSCHAFT_PYTHON "Build the Python module landschaft._core" ON)
# The MILP prioritization (optcore, with HiGHS fetched and built): always with Python, on demand without it.
include(CMakeDependentOption)
cmake_dependent_option(LANDSCHAFT_MILP "Build the MILP prioritization (optcore and HiGHS)" OFF
                       "NOT LANDSCHAFT_PYTHON" ON)

# On Windows the modules are built with MSVC (the results are checked against the UCRT's maths);
# a MinGW gcc found on PATH would make modules that need its runtime DLLs and fail to import.
if(WIN32 AND NOT MSVC)
    message(FATAL_ERROR "landschaft is built with MSVC on Windows: install Visual Studio (or its Build Tools) "
                        "with the C++ workload. The compiler found was ${CMAKE_CXX_COMPILER}.")
endif()

if(LANDSCHAFT_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
endif()

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
    src/cpp/nlm/basic.cpp
    src/cpp/nlm/cluster.cpp
    src/cpp/nlm/distance.cpp
    src/cpp/nlm/mosaic.cpp
    src/cpp/nlm/rfft.cpp
    src/cpp/nlm/rng.cpp
    src/cpp/nlm/spectral.cpp
    src/cpp/nlm/stdmodels.cpp
    src/cpp/nlm/tess.cpp
)
target_include_directories(nlmcore PUBLIC src/cpp)
# R's x87 log (lsm/rmath.hpp).
target_link_libraries(nlmcore PUBLIC lsmcore)
set_target_properties(nlmcore PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Distances and routes on grids (gdistance): transitions, least-cost and random-walk distances.
add_library(costcore STATIC
    src/cpp/cost/frontal_lu.cpp
    src/cpp/cost/geodesic.cpp
    src/cpp/cost/randomwalk.cpp
    src/cpp/cost/rsp.cpp
    src/cpp/cost/shortest.cpp
    src/cpp/cost/transition.cpp
)
target_include_directories(costcore PUBLIC src/cpp)
# R's x87 maths and 80-bit sums (lsm/rmath.hpp, lsm/extended.hpp); the
# circuit module's multifrontal Cholesky for the random walks.
target_link_libraries(costcore PUBLIC lsmcore circuitcore)
target_link_libraries(costcore PRIVATE eigen_headers)
set_target_properties(costcore PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Biodiversity analyses (Biodiverse): spatial conditions, indices, randomisations.
add_library(biocore STATIC
    src/cpp/bio/conditions.cpp
    src/cpp/bio/indices.cpp
    src/cpp/bio/phylo.cpp
    src/cpp/bio/phylocom.cpp
    src/cpp/bio/props.cpp
    src/cpp/bio/randomise.cpp
    src/cpp/bio/cluster.cpp
)
target_include_directories(biocore PUBLIC src/cpp)
# Perl's maths on Windows is mingw's x87 libm (lsm/rmath.hpp).
target_link_libraries(biocore PUBLIC lsmcore)
set_target_properties(biocore PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Surface metrics (geodiv's ISO 25178 parameters, spatialEco's terrain indices).
add_library(surfcore STATIC
    src/cpp/surf/amplitude.cpp
    src/cpp/surf/bearing.cpp
    src/cpp/surf/features.cpp
    src/cpp/surf/geomorphons.cpp
    src/cpp/surf/gravity.cpp
    src/cpp/surf/hilltop.cpp
    src/cpp/surf/hybrid.cpp
    src/cpp/surf/plane.cpp
    src/cpp/surf/spatial.cpp
    src/cpp/surf/terrain.cpp
    src/cpp/surf/window.cpp
)
target_include_directories(surfcore PUBLIC src/cpp)
# R's x87 maths and 80-bit sums (lsm/rmath.hpp, lsm/extended.hpp), R's fft (nlm/rfft.hpp), geodesics (cost/geodesic.hpp).
target_link_libraries(surfcore PUBLIC lsmcore nlmcore costcore)
set_target_properties(surfcore PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Terrain hydrology: depression filling, flow directions and accumulation, HAND.
add_library(hydrocore STATIC
    src/cpp/hydro/basins.cpp
    src/cpp/hydro/breach.cpp
    src/cpp/hydro/hydro.cpp
    src/cpp/hydro/mfd.cpp
)
target_include_directories(hydrocore PUBLIC src/cpp)
set_target_properties(hydrocore PROPERTIES POSITION_INDEPENDENT_CODE ON)
# SWAT+ 62.0.1 (the ls.hydro.swat subset): float arithmetic as the Fortran reference.
add_library(swatcore STATIC
    src/cpp/swat/aquifer.cpp
    src/cpp/swat/calibration.cpp
    src/cpp/swat/channel.cpp
    src/cpp/swat/climate.cpp
    src/cpp/swat/command.cpp
    src/cpp/swat/connect.cpp
    src/cpp/swat/fio.cpp
    src/cpp/swat/hru_daily.cpp
    src/cpp/swat/hru_init.cpp
    src/cpp/swat/mgt.cpp
    src/cpp/swat/output.cpp
    src/cpp/swat/read_basin.cpp
    src/cpp/swat/read_db.cpp
    src/cpp/swat/time_control.cpp
)
target_include_directories(swatcore PUBLIC src/cpp)
set_target_properties(swatcore PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Spatial prioritization (Zonation 5): quantized feature data and the iterative ranking.
add_library(priocore STATIC
    src/cpp/prio/buffer.cpp
    src/cpp/prio/data.cpp
    src/cpp/prio/rank.cpp
)
target_include_directories(priocore PUBLIC src/cpp)
# Eigen's generic_pow on Packet2d (Zonation's Array2d::pow).
target_link_libraries(priocore PRIVATE eigen_headers)
set_target_properties(priocore PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(LANDSCHAFT_MILP)
    # HiGHS 1.15.1 (MIT), the MILP / LP solver of the MILP prioritization, built
    # here as a static library linked into _core (no executable, tests or zlib).
    # FETCHCONTENT_SOURCE_DIR_HIGHS=<path> uses a local copy (offline builds).
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)  # HiGHS's option() calls take these values
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_CXX_EXE OFF)
    set(BUILD_TESTING OFF)
    set(BUILD_EXAMPLES OFF)
    set(BUILD_SHARED_EXTRAS_LIB OFF)
    set(ZLIB OFF)
    set(FORTRAN OFF)
    set(CSHARP OFF)
    set(HIPO OFF)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    # the script's hash in the patch command: a changed script patches a fetched copy again (it skips what is done)
    file(MD5 ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/opt/highs_patch.cmake highs_patch_md5)
    FetchContent_Declare(highs
        URL https://github.com/ERGO-Code/HiGHS/archive/refs/tags/v1.15.1.tar.gz
        URL_HASH SHA256=a840d269dff2fafb371dd247df13ad5e026d7ce3b35ad3dc1eedd59bf0c2fb16
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        # our changes to its sources (see the script)
        PATCH_COMMAND ${CMAKE_COMMAND} -DPATCH_MD5=${highs_patch_md5} -P ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/opt/highs_patch.cmake)
    FetchContent_MakeAvailable(highs)
    unset(CMAKE_POSITION_INDEPENDENT_CODE)
    unset(BUILD_SHARED_LIBS)

    # MILP prioritization (prioritizr / prioriactions): the model compiler, our own
    # presolve and heuristics, and the calls to HiGHS.
    add_library(optcore STATIC
        src/cpp/opt/actions.cpp
        src/cpp/opt/closure.cpp
        src/cpp/opt/connect.cpp
        src/cpp/opt/cover.cpp
        src/cpp/opt/cut.cpp
        src/cpp/opt/eval.cpp
        src/cpp/opt/features.cpp
        src/cpp/opt/marxan.cpp
        src/cpp/opt/maxflow.cpp
        src/cpp/opt/milp.cpp
        src/cpp/opt/patch.cpp
        src/cpp/opt/proxirr.cpp
        src/cpp/opt/wide.cpp
        src/cpp/opt/zones.cpp
        src/cpp/opt/zones_wide.cpp
        src/cpp/opt/zones_cut.cpp
    )
    target_include_directories(optcore PUBLIC src/cpp)
    target_include_directories(optcore SYSTEM PRIVATE ${highs_SOURCE_DIR}/highs ${highs_BINARY_DIR})
    target_link_libraries(optcore PRIVATE highs)
    # R's x87 exp (lsm/rmath.hpp) in the evaluation's pnorm
    target_link_libraries(optcore PUBLIC lsmcore)
    set_target_properties(optcore PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
# Species distribution modelling: thinning, background sampling, partitions, evaluation, MaxNet.
add_library(sdmcore STATIC
    src/cpp/sdm/clean.cpp
    src/cpp/sdm/collinear.cpp
    src/cpp/sdm/extrapolation.cpp
    src/cpp/sdm/geocircle/geocircle.cpp
    src/cpp/sdm/geocircle/geocircle_geodesic.cpp
    src/cpp/sdm/geocircle/geos_lite.cpp
    src/cpp/sdm/lognet.cpp
    src/cpp/sdm/maxnet.cpp
    src/cpp/sdm/rstats.cpp
    src/cpp/sdm/kmeans.cpp
    src/cpp/sdm/rformat.cpp
    src/cpp/sdm/thin.cpp
    src/cpp/sdm/variogram.cpp
)
target_include_directories(sdmcore PUBLIC src/cpp)
# R's x87 maths (lsm/rmath.hpp), R's generator (nlm/rng.hpp), mingw's cbrt (cost/geodesic.hpp).
target_link_libraries(sdmcore PUBLIC lsmcore nlmcore costcore)
set_target_properties(sdmcore PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Restoration planning inputs (after plangea): original vegetation, area of habitat, ecological intactness.
add_library(restcore STATIC
    src/cpp/rest/habitat.cpp
    src/cpp/rest/intactness.cpp
    src/cpp/rest/plan.cpp
)
target_include_directories(restcore PUBLIC src/cpp)
# R's x87 maths and 80-bit sums (lsm/rmath.hpp, lsm/extended.hpp).
target_link_libraries(restcore PUBLIC lsmcore)
set_target_properties(restcore PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Ecosystem services (InVEST): carbon storage, habitat quality.
add_library(servcore STATIC
    src/cpp/serv/ndr.cpp
    src/cpp/serv/pollination.cpp
    src/cpp/serv/sdr.cpp
    src/cpp/serv/services.cpp
    src/cpp/serv/swy.cpp
)
target_include_directories(servcore PUBLIC src/cpp)
set_target_properties(servcore PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Land use change (ls.simulate): cross-tabulation and change counts.
add_library(simcore STATIC
    src/cpp/sim/change.cpp
    src/cpp/sim/woe.cpp
    src/cpp/sim/rglm.cpp
    src/cpp/sim/clues.cpp
    src/cpp/sim/clumondo.cpp
    src/cpp/sim/continuous.cpp
    src/cpp/sim/demand.cpp
    src/cpp/sim/futures.cpp
    src/cpp/sim/molusce.cpp
    src/cpp/sim/observed.cpp
    src/cpp/sim/ordered.cpp
    src/cpp/sim/patches.cpp
    src/cpp/sim/plus.cpp
    src/cpp/sim/roads.cpp
    src/cpp/sim/rules.cpp
    src/cpp/sim/similarity.cpp
    src/cpp/sim/validation.cpp
)
target_include_directories(simcore PUBLIC src/cpp)
target_link_libraries(simcore PUBLIC lsmcore nlmcore)
# the matrix logarithm and exponential of the Markov chains (unsupported/Eigen/MatrixFunctions)
target_link_libraries(simcore PRIVATE eigen_headers)
set_target_properties(simcore PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Raster values by zone and at points (ls.zonal_stats, ls.extract), ls.reclassify, GDAL's warp
# kernel (ls.reproject); GDAL's polygon rasterizer (lsm/rasterize.cpp); exactextract's exact cell
# coverage of polygons (src/cpp/io/exact, Apache-2.0).
add_library(iocore STATIC
    src/cpp/io/exact.cpp
    src/cpp/io/exact/box.cpp
    src/cpp/io/exact/cell.cpp
    src/cpp/io/exact/coordinate.cpp
    src/cpp/io/exact/coverage.cpp
    src/cpp/io/exact/grid.cpp
    src/cpp/io/exact/measures.cpp
    src/cpp/io/exact/perimeter_distance.cpp
    src/cpp/io/exact/side.cpp
    src/cpp/io/exact/traversal.cpp
    src/cpp/io/exact/traversal_areas.cpp
    src/cpp/io/reclass.cpp
    src/cpp/io/triangulate.cpp
    src/cpp/io/view.cpp
    src/cpp/io/warp.cpp
    src/cpp/io/zonal.cpp
)
target_include_directories(iocore PUBLIC src/cpp)
target_link_libraries(iocore PUBLIC lsmcore)
set_target_properties(iocore PROPERTIES POSITION_INDEPENDENT_CODE ON)
# R's x87 log/sqrt and LDOUBLE sums (see src/cpp/lsm/rmath.hpp): MSVC has no
# x64 inline assembly. x86-64 only; every other target uses the C++ code.
if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|x86_64)$")
    enable_language(ASM_MASM)
    target_sources(lsmcore PRIVATE src/cpp/lsm/x87_msvc.asm)
endif()
set_target_properties(lsmcore PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(gtbcore PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(LANDSCHAFT_OPENMP)
    # Apple's clang has no OpenMP runtime of its own and only conda's CMake
    # looks in the environment for one: when building for a conda Python,
    # take its environment's (llvm-openmp), else Homebrew's libomp, unless
    # OpenMP_ROOT is given. The conda one first, as other packages of that
    # environment may load it too (two libomp in one process abort).
    if(APPLE AND NOT DEFINED OpenMP_ROOT AND NOT DEFINED ENV{OpenMP_ROOT})
        set(_conda_python FALSE)
        if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/include/omp.h")
            set(_conda_prefix "$ENV{CONDA_PREFIX}")
            cmake_path(IS_PREFIX _conda_prefix "${Python_EXECUTABLE}" NORMALIZE _conda_python)
        endif()
        if(_conda_python)
            set(OpenMP_ROOT "$ENV{CONDA_PREFIX}")
        else()
            find_program(LANDSCHAFT_BREW brew)
            if(LANDSCHAFT_BREW)
                execute_process(COMMAND "${LANDSCHAFT_BREW}" --prefix libomp
                                OUTPUT_VARIABLE _libomp_prefix OUTPUT_STRIP_TRAILING_WHITESPACE
                                RESULT_VARIABLE _brew_result ERROR_QUIET)
                if(_brew_result EQUAL 0 AND EXISTS "${_libomp_prefix}/include/omp.h")
                    set(OpenMP_ROOT "${_libomp_prefix}")
                endif()
            endif()
        endif()
        if(DEFINED OpenMP_ROOT)
            message(STATUS "OpenMP runtime from ${OpenMP_ROOT}")
        endif()
    endif()
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(gtbcore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(lsmcore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(conncore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(nlmcore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(biocore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(costcore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(circuitcore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(surfcore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(hydrocore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(priocore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(iocore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(sdmcore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(simcore PUBLIC OpenMP::OpenMP_CXX)
        if(TARGET optcore)
            target_link_libraries(optcore PUBLIC OpenMP::OpenMP_CXX)
        endif()
        target_link_libraries(restcore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(servcore PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(swatcore PUBLIC OpenMP::OpenMP_CXX)
        # conda's libomp is found through @rpath: keep its directory in the
        # installed modules (created below) too, not only in the build tree.
        if(APPLE)
            set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
        endif()
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
    target_compile_options(biocore PRIVATE /fp:precise /W4 /permissive-)
    target_compile_options(costcore PRIVATE /fp:precise /W4 /permissive-)
    target_compile_options(circuitcore PRIVATE /fp:precise /W4 /permissive-)
    target_compile_options(surfcore PRIVATE /fp:precise /W4 /permissive-)
    target_compile_options(hydrocore PRIVATE /fp:precise /W4 /permissive-)
    target_compile_options(priocore PRIVATE /fp:precise /W4 /permissive-)
    target_compile_options(iocore PRIVATE /fp:precise /W4 /permissive-)
    target_compile_options(sdmcore PRIVATE /fp:precise /W4 /permissive-)
    if(TARGET optcore)
        target_compile_options(optcore PRIVATE /fp:precise /W4 /permissive-)
    endif()
    target_compile_options(simcore PRIVATE /fp:precise /W4 /permissive-)
    target_compile_options(restcore PRIVATE /fp:precise /W4 /permissive-)
    target_compile_options(servcore PRIVATE /fp:precise /W4 /permissive-)
    target_compile_options(swatcore PRIVATE /fp:precise /W4 /permissive-)
else()
    target_compile_options(gtbcore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    target_compile_options(lsmcore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    target_compile_options(conncore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    target_compile_options(nlmcore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    target_compile_options(biocore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    target_compile_options(costcore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    target_compile_options(circuitcore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    target_compile_options(surfcore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    target_compile_options(hydrocore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    target_compile_options(priocore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    target_compile_options(iocore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    target_compile_options(sdmcore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    if(TARGET optcore)
        target_compile_options(optcore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    endif()
    target_compile_options(simcore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    target_compile_options(restcore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    target_compile_options(servcore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra)
    # a double literal or promotion in the float code would change its bits
    target_compile_options(swatcore PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra -Wdouble-promotion)
endif()

if(LANDSCHAFT_PYTHON)
    pybind11_add_module(_core MODULE src/cpp/bindings.cpp src/cpp/circuit/python.cpp src/cpp/nlm/python.cpp
                        src/cpp/bio/python.cpp src/cpp/cost/python.cpp src/cpp/surf/python.cpp
                        src/cpp/prio/python.cpp src/cpp/io/python.cpp src/cpp/sdm/python.cpp src/cpp/sdm/maxnet_python.cpp
                        src/cpp/sim/python.cpp src/cpp/opt/python.cpp src/cpp/rest/python.cpp src/cpp/hydro/python.cpp
                        src/cpp/serv/python.cpp src/cpp/serv/python_swy.cpp
                        src/cpp/swat/python.cpp)
    target_link_libraries(_core PRIVATE eigen_headers)
    target_link_libraries(_core PRIVATE gtbcore lsmcore conncore circuitcore nlmcore biocore costcore surfcore priocore
                          iocore sdmcore simcore optcore restcore hydrocore servcore swatcore)
    if(MSVC)
        target_compile_options(_core PRIVATE /fp:precise)
    else()
        target_compile_options(_core PRIVATE -ffp-contract=off -fno-fast-math)
    endif()

    # Only this component goes into the wheel (not the headers and library HiGHS installs).
    install(TARGETS _core DESTINATION landschaft COMPONENT landschaft)

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
        install(TARGETS _circuit_avx2 DESTINATION landschaft COMPONENT landschaft)
    endif()
else()
    # The static libraries and their headers, for native programs (Terraflux links simcore and what it needs).
    install(TARGETS gtbcore lsmcore conncore circuitcore nlmcore biocore costcore surfcore priocore iocore sdmcore
                    simcore restcore hydrocore servcore swatcore
            ARCHIVE DESTINATION lib)
    # with HiGHS's library, which its own rules install
    if(LANDSCHAFT_MILP)
        install(TARGETS optcore ARCHIVE DESTINATION lib)
    endif()
    install(DIRECTORY src/cpp/ DESTINATION include FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" PATTERN "*.inc")
    install(DIRECTORY ${eigen_SOURCE_DIR}/Eigen ${eigen_SOURCE_DIR}/unsupported DESTINATION include)
endif()
