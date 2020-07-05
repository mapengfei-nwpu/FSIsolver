cmake_minimum_required(VERSION 3.5)
project(test_function)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find dolfin
find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

# Add pybin11 module
pybind11_add_module(IBMesh SHARED IBMesh.cpp)
pybind11_add_module(IBInterpolation SHARED IBInterpolation.cpp)

# Target the library
target_link_libraries(IBMesh PRIVATE dolfin)
target_link_libraries(IBInterpolation PRIVATE dolfin)

