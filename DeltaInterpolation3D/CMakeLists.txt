cmake_minimum_required(VERSION 3.5)
project(test_function)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find dolfin
find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

# add_definitions("-Wall -g")
add_definitions("-O3")

# Add pybin11 module
pybind11_add_module(IB SHARED IBMesh.cpp IBInterpolation.cpp IB.cpp LocalPolynomial.cpp)

# Target the library
target_link_libraries(IB PRIVATE dolfin)
