# Top-level build for the faster-enhancer-py ctypes binding.
#
# Compiles the vendored C engine (csrc/) into a single shared library
# (fe_shared) that the Python layer loads via ctypes. The source globs and
# per-ISA -march flags mirror the upstream CMakeLists exactly; the differences
# are intentional for a ctypes-loadable artifact:
#   - SHARED, not STATIC
#   - POSITION_INDEPENDENT_CODE ON
#   - no -fvisibility=hidden  (so the fe_* ABI is exported for ctypes; the C
#     engine has no other public surface, so exporting all symbols is harmless)
#   - WINDOWS_EXPORT_ALL_SYMBOLS ON for Windows
cmake_minimum_required(VERSION 3.20)
project(faster_enhancer_py LANGUAGES C)

set(CSRC ${CMAKE_CURRENT_SOURCE_DIR}/csrc)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Arch detection (same matrix as upstream: arm64 + x86_64). ---
if(MSVC)
    message(FATAL_ERROR "fe requires GCC/Clang target attributes; use MinGW/clang on Windows, not MSVC.")
endif()
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _proc)
if(_proc MATCHES "aarch64|arm64|armv8")
    set(FE_ARCH "arm64")
elseif(_proc MATCHES "x86_64|amd64|x64")
    set(FE_ARCH "x86_64")
else()
    set(FE_ARCH "unknown")
endif()
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    list(LENGTH CMAKE_OSX_ARCHITECTURES _osx_n)
    if(_osx_n EQUAL 1)
        if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
            set(FE_ARCH "arm64")
        elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
            set(FE_ARCH "x86_64")
        endif()
    endif()
endif()
if(NOT (FE_ARCH STREQUAL "arm64" OR FE_ARCH STREQUAL "x86_64"))
    message(FATAL_ERROR "fe supports arm64 and x86_64 only (got '${CMAKE_SYSTEM_PROCESSOR}').")
endif()
message(STATUS "faster-enhancer-py: arch=${FE_ARCH}")

set(FE_FLAGS_COMMON
    -O3 -Wall -Wextra -Wno-unused-parameter
    -fno-math-errno -fno-trapping-math -ffinite-math-only
    -fmerge-all-constants -fno-exceptions
    -fno-stack-protector
    -DNDEBUG)

# --- Engine sources (upstream globs). ---
file(GLOB FE_SRC_TOP   CONFIGURE_DEPENDS ${CSRC}/src/*.c)
file(GLOB FE_SRC_NN    CONFIGURE_DEPENDS ${CSRC}/src/nn/*.c)
file(GLOB FE_SRC_QGEMM CONFIGURE_DEPENDS ${CSRC}/src/qgemm/*.c)
file(GLOB FE_SRC_WINO  CONFIGURE_DEPENDS ${CSRC}/src/winograd/*.c)
set(FE_SRC ${FE_SRC_TOP} ${FE_SRC_NN} ${FE_SRC_QGEMM} ${FE_SRC_WINO})
if(FE_ARCH STREQUAL "arm64")
    file(GLOB FE_SRC_ARM CONFIGURE_DEPENDS ${CSRC}/src/qgemm/arm/*.c)
    list(APPEND FE_SRC ${FE_SRC_ARM} ${CSRC}/src/fft/fft_neon.c)
elseif(FE_ARCH STREQUAL "x86_64")
    file(GLOB FE_SRC_X86 CONFIGURE_DEPENDS ${CSRC}/src/qgemm/x86/*.c)
    list(APPEND FE_SRC ${FE_SRC_X86}
                       ${CSRC}/src/fft/fft_avx2.c
                       ${CSRC}/src/fft/fft_avx512.c)
endif()

add_library(fe_shared SHARED ${FE_SRC})
target_include_directories(fe_shared
    PUBLIC  ${CSRC}/include
    PRIVATE ${CSRC}/src ${CSRC}/src/internal)
target_compile_options(fe_shared PRIVATE ${FE_FLAGS_COMMON})
set_target_properties(fe_shared PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    OUTPUT_NAME fe
    SOVERSION 0)
if(NOT WIN32)
    target_link_libraries(fe_shared PRIVATE m)
endif()

# --- Per-file -march flags for each ISA tier (verbatim from upstream). ---
if(FE_ARCH STREQUAL "arm64")
    set_source_files_properties(${CSRC}/src/qgemm/arm/qgemm_dotprod.c PROPERTIES
        COMPILE_FLAGS "-march=armv8.2-a+dotprod")
    set_source_files_properties(${CSRC}/src/qgemm/arm/qgemm_i8mm.c PROPERTIES
        COMPILE_FLAGS "-march=armv8.6-a+dotprod+i8mm")
elseif(FE_ARCH STREQUAL "x86_64")
    set_source_files_properties(${CSRC}/src/qgemm/x86/qgemm_avx2.c PROPERTIES
        COMPILE_FLAGS "-mavx2 -mfma")
    set_source_files_properties(${CSRC}/src/fft/fft_avx2.c PROPERTIES
        COMPILE_FLAGS "-mavx2 -mfma")
    set_source_files_properties(${CSRC}/src/qgemm/x86/qgemm_avxvnni.c PROPERTIES
        COMPILE_FLAGS "-mavx2 -mfma -mavxvnni")
    set_source_files_properties(${CSRC}/src/qgemm/x86/qgemm_avx512vnni.c PROPERTIES
        COMPILE_FLAGS "-mavx2 -mfma -mavxvnni -mavx512f -mavx512bw -mavx512vl -mavx512vnni")
    set_source_files_properties(${CSRC}/src/fft/fft_avx512.c PROPERTIES
        COMPILE_FLAGS "-mavx2 -mfma -mavxvnni -mavx512f -mavx512bw -mavx512vl -mavx512vnni")
    set(FE_X86_AVX2_TUS
        ${CSRC}/src/winograd/fe_winograd.c
        ${CSRC}/src/qgemm/qgemm_quant.c
        ${CSRC}/src/nn/fe_activations.c
        ${CSRC}/src/nn/fe_attention.c
        ${CSRC}/src/fe_engine.c
        ${CSRC}/src/nn/fe_gru.c
        ${CSRC}/src/nn/fe_qgemm.c
        ${CSRC}/src/nn/fe_sgemm.c
        ${CSRC}/src/fe_stft.c
        ${CSRC}/src/nn/fe_vec.c)
    set_source_files_properties(${FE_X86_AVX2_TUS} PROPERTIES
        COMPILE_FLAGS "-mavx2 -mfma -mf16c")
endif()

# Install the shared library next to the Python package so ctypes can find it
# without fighting RPATH: <package>/faster_enhancer/libfe.{so,dylib,dll}.
install(TARGETS fe_shared
        LIBRARY DESTINATION faster_enhancer
        RUNTIME DESTINATION faster_enhancer
        ARCHIVE DESTINATION faster_enhancer)
# Ship the vendored weights blob alongside the binding.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/faster_enhancer/weights/fe.q8
        DESTINATION faster_enhancer/weights)
