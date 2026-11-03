
cmake_minimum_required(VERSION 3.20)

project(Mini-COLMAP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSYS2 UCRT64 dependencies
list(PREPEND CMAKE_PREFIX_PATH
        "C:/msys64/ucrt64"
)

# OpenCV
set(OpenCV_DIR
        "C:/msys64/ucrt64/lib/cmake/opencv5"
)

find_package(OpenCV REQUIRED)

# SuiteSparse
set(SUITESPARSE_INCLUDE_DIR_HINTS
        "C:/msys64/ucrt64/include/suitesparse"
)

set(SUITESPARSE_LIBRARY_DIR_HINTS
        "C:/msys64/ucrt64/lib"
)

# Ceres Solver
find_package(Ceres REQUIRED)

# Executable
add_executable(Mini-COLMAP
        main.cpp
        src/FeatureExtractor.cpp
        src/FeatureMatcher.cpp
        src/Geometry.cpp
        src/Triangulation.cpp
        src/Reconstruction.cpp
        src/BundleAdjuster.cpp
        src/Diagnostics.cpp
        src/Retrieval.cpp
)

# Compiler options
target_compile_options(Mini-COLMAP PRIVATE
        -Wall
        -Wextra
        -Wreturn-type
)

# Linker options
target_link_options(Mini-COLMAP PRIVATE
        -static-libstdc++
        -static-libgcc
)

# Include directories
target_include_directories(Mini-COLMAP PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(Mini-COLMAP PRIVATE
        ${OpenCV_LIBS}
        Ceres::ceres
)
